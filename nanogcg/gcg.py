import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from tqdm import tqdm
from transformers import set_seed

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    get_nonascii_toks,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: str = "You are a helpful assistant."
    search_width: int = 64
    search_batch_size: Optional[int] = None
    dataset_batch_size: Optional[int] = None
    gradient_sample_size: Optional[int] = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    fluency_weight: float = 0.0
    allow_non_ascii: bool = False
    filter_ids: bool = True
    seed: Optional[int] = None
    verbosity: str = "INFO"


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            s = tokenizer.batch_decode(ids)[0].replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\nloss: {loss} | string: {s}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
):
    """Returns `search_width` combinations of token ids based on the token gradient."""
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    return original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Drop sequences whose token ids change after decode+re-encode."""
    decoded = tokenizer.batch_decode(ids)
    kept = []
    for i in range(len(decoded)):
        encoded = tokenizer(decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], encoded):
            kept.append(ids[i])
    if not kept:
        raise RuntimeError(
            "All candidate token sequences changed under decode/re-encode. "
            "Try `filter_ids=False` or a different `optim_str_init`."
        )
    return torch.stack(kept)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision dtype for faster optimization.")
        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")
        if not tokenizer.chat_template:
            logger.warning("Tokenizer has no chat template. Using a passthrough template.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        if tokenizer.pad_token is None:
            configure_pad_token(tokenizer)

    def run(self, system_template: str, completion_dataset: list[dict]) -> GCGResult:
        config = self.config
        model = self.model
        tokenizer = self.tokenizer
        device = model.device

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        assert "{optim_str}" in system_template, "system_template must contain the literal '{optim_str}' placeholder"

        # Build the templated text for every example, splitting on the placeholder.
        before_strs, after_strs, target_strs = [], [], []
        for ex in completion_dataset:
            messages = [
                {"role": "system", "content": system_template},
                {"role": "user", "content": ex["prompt"]},
            ]
            templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if tokenizer.bos_token and templated.startswith(tokenizer.bos_token):
                templated = templated[len(tokenizer.bos_token):]
            before, after = templated.split("{optim_str}")
            before_strs.append(before)
            after_strs.append(after)
            target_strs.append(ex["completion"])

        assert all(b == before_strs[0] for b in before_strs), "before-string differs across the dataset; chat template is not stable across examples"
        before_str = before_strs[0]

        # Fix the optim length once from the seed.
        init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        optim_len = init_optim_ids.shape[1]

        before_ids = tokenizer([before_str], return_tensors="pt")["input_ids"][0].to(device, torch.int64)
        after_lists = [tokenizer(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device) for a in after_strs]
        target_lists = [tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device) for t in target_strs]

        N = len(completion_dataset)
        after_lens = torch.tensor([len(a) for a in after_lists], device=device)
        target_lens = torch.tensor([len(t) for t in target_lists], device=device)
        combined_lens = after_lens + target_lens
        max_combined = int(combined_lens.max())

        pad_id = tokenizer.pad_token_id
        combined_ids = torch.full((N, max_combined), pad_id, dtype=torch.int64, device=device)
        for i in range(N):
            al = len(after_lists[i])
            tl = len(target_lists[i])
            combined_ids[i, :al] = after_lists[i]
            combined_ids[i, al:al + tl] = target_lists[i]

        before_len = before_ids.shape[0]
        seq_len = before_len + optim_len + max_combined

        self.before_embeds = self.embedding_layer(before_ids).unsqueeze(0)
        self.combined_embeds = self.embedding_layer(combined_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_lens = before_len + optim_len + combined_lens
        self.attention_mask = (positions < valid_lens.unsqueeze(1)).to(torch.int64)

        labels = torch.full((N, seq_len), -100, dtype=torch.int64, device=device)
        for i in range(N):
            al = int(after_lens[i].item())
            tl = int(target_lens[i].item())
            t_start = before_len + optim_len + al
            labels[i, t_start:t_start + tl] = combined_ids[i, al:al + tl]
        self.labels = labels

        self.before_len = before_len
        self.optim_len = optim_len
        self.seq_len = seq_len
        self.N = N

        full_indices = torch.arange(N, device=device)
        buffer = self.init_buffer(init_optim_ids, full_indices)
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for _ in tqdm(range(config.num_steps), desc="gcg", ascii=" >="):
            if config.gradient_sample_size is None or config.gradient_sample_size >= N:
                indices = full_indices
            else:
                indices = torch.randperm(N, device=device)[:config.gradient_sample_size]

            grad = self.compute_token_gradient(optim_ids, indices)

            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )
                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                cand_losses = self.compute_candidates_loss(sampled_ids, indices)
                best_idx = cand_losses.argmin()
                current_loss = cand_losses[best_idx].item()
                optim_ids = sampled_ids[best_idx].unsqueeze(0)

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_strings.append(tokenizer.batch_decode(optim_ids)[0])
            buffer.log_buffer(tokenizer)

        min_idx = losses.index(min(losses))
        return GCGResult(
            best_loss=losses[min_idx],
            best_string=optim_strings[min_idx],
            losses=losses,
            strings=optim_strings,
        )

    def init_buffer(self, init_optim_ids: Tensor, indices: Tensor) -> AttackBuffer:
        config = self.config
        tokenizer = self.tokenizer
        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")
        buffer = AttackBuffer(config.buffer_size)

        if config.buffer_size > 1:
            init_chars_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(self.model.device)
            rand_idx = torch.randint(0, init_chars_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
            init_buffer_ids = torch.cat([init_optim_ids, init_chars_ids[rand_idx]], dim=0)
        else:
            init_buffer_ids = init_optim_ids

        true_size = max(1, config.buffer_size)
        init_losses = self.compute_candidates_loss(init_buffer_ids, indices)
        for i in range(true_size):
            buffer.add(init_losses[i].item(), init_buffer_ids[[i]])
        buffer.log_buffer(tokenizer)
        return buffer

    def _forward_example_minibatch(self, optim_embeds: Tensor, indices: Tensor) -> Tensor:
        """Forward `eb = len(indices)` examples sharing `optim_embeds`. Returns per-example loss (eb,)."""
        eb = indices.shape[0]
        bs = optim_embeds.shape[0]
        if bs == 1 and eb > 1:
            optim_embeds_b = optim_embeds.expand(eb, -1, -1)
        else:
            optim_embeds_b = optim_embeds
        assert optim_embeds_b.shape[0] == eb

        input_embeds = torch.cat([
            self.before_embeds.expand(eb, -1, -1),
            optim_embeds_b,
            self.combined_embeds[indices],
        ], dim=1)
        attention_mask = self.attention_mask[indices]
        labels = self.labels[indices]

        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(eb, -1)
        valid = (shift_labels != -100).to(token_loss.dtype)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def compute_token_gradient(self, optim_ids: Tensor, indices: Tensor) -> Tensor:
        """Mean-loss gradient over `indices` w.r.t. the one-hot of `optim_ids`. Accumulated over minibatches."""
        config = self.config
        model = self.model
        K = indices.shape[0]
        ex_bs = K if config.dataset_batch_size is None else config.dataset_batch_size

        optim_ids_onehot = F.one_hot(optim_ids, num_classes=self.embedding_layer.num_embeddings).to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        for start in range(0, K, ex_bs):
            end = min(start + ex_bs, K)
            chunk = indices[start:end]
            optim_embeds = optim_ids_onehot @ self.embedding_layer.weight
            per_example = self._forward_example_minibatch(optim_embeds, chunk)
            (per_example.sum() / K).backward()

        if config.fluency_weight > 0 and self.optim_len >= 2:
            optim_embeds = optim_ids_onehot @ self.embedding_layer.weight
            out = model(inputs_embeds=optim_embeds)
            sh = out.logits[:, :-1, :].contiguous()
            lb = optim_ids[:, 1:].contiguous()
            nll = F.cross_entropy(sh.view(-1, sh.size(-1)), lb.view(-1))
            (config.fluency_weight * nll).backward()

        return optim_ids_onehot.grad.detach().clone()

    def compute_candidates_loss(self, candidate_ids: Tensor, indices: Tensor) -> Tensor:
        """Mean loss over `indices` for each candidate. Returns (n_candidates,)."""
        config = self.config
        model = self.model
        n = candidate_ids.shape[0]
        K = indices.shape[0]
        ex_bs = K if config.dataset_batch_size is None else config.dataset_batch_size

        losses = torch.zeros(n, device=model.device, dtype=torch.float32)
        with torch.no_grad():
            for c in range(n):
                cand_embeds = self.embedding_layer(candidate_ids[c:c + 1])
                acc = torch.zeros((), device=model.device)
                for start in range(0, K, ex_bs):
                    end = min(start + ex_bs, K)
                    chunk = indices[start:end]
                    per_example = self._forward_example_minibatch(cand_embeds, chunk)
                    acc = acc + per_example.sum()
                losses[c] = (acc / K).to(losses.dtype)

            if config.fluency_weight > 0 and self.optim_len >= 2:
                cand_embeds_all = self.embedding_layer(candidate_ids)
                fl_logits = model(inputs_embeds=cand_embeds_all).logits
                sh = fl_logits[:, :-1, :].contiguous()
                lb = candidate_ids[:, 1:].contiguous()
                fl = F.cross_entropy(
                    sh.reshape(-1, sh.size(-1)),
                    lb.reshape(-1),
                    reduction="none",
                ).view(n, -1).mean(dim=1)
                losses += config.fluency_weight * fl.to(losses.dtype)

        return losses


def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    system_template: str,
    completion_dataset: list[dict],
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Optimize a system prompt to maximize likelihood of `completion_dataset` under `model`.

    Args:
        model: causal LM (HF).
        tokenizer: matching tokenizer.
        system_template: string containing the literal "{optim_str}" placeholder; optimized tokens go in its place.
        completion_dataset: list of {"prompt": str, "completion": str} dicts.
        config: optional GCGConfig.
    """
    if config is None:
        config = GCGConfig()
    logger.setLevel(getattr(logging, config.verbosity))
    return GCG(model, tokenizer, config).run(system_template, completion_dataset)
