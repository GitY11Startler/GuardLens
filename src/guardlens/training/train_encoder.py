"""Fine-tune the D1 encoder: a sequence classifier over benign + the seen attack families.

Plain PyTorch loop (no `Trainer`): AdamW, linear warmup then linear decay, bf16 autocast on
CUDA, gradient accumulation. After every epoch the model scores dev; the epoch with the best
dev AUROC is kept. The temperature for calibration is then fitted on that epoch's dev scores.

Run folder (`data/models/d1/<run_name>/`):
    model/              best checkpoint (weights + tokenizer)
    config.json         D1 config + git hash + data manifest hash
    history.json        per-epoch train loss and dev metrics
    calibration.json    temperature + ECE/Brier before and after
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from guardlens.data.schema import ATTACK, Example
from guardlens.eval.config import D1Config
from guardlens.eval.metrics import auroc
from guardlens.run_info import git_commit
from guardlens.training.calibrate import (
    attack_logit,
    brier,
    calibrated_score,
    ece,
    fit_temperature,
    log_loss,
    plot_reliability,
    sigmoid,
)


def special_tokens(tokenizer: PreTrainedTokenizerBase) -> tuple[list[int], list[int]]:
    """The special token ids the tokenizer puts before and after a single text
    (e.g. [CLS] ... [SEP]), found by tokenizing a probe with and without them."""
    probe = "placeholder"
    plain = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    full = list(tokenizer(probe)["input_ids"])
    for i in range(len(full) - len(plain) + 1):
        if full[i : i + len(plain)] == plain:
            return full[:i], full[i + len(plain) :]
    raise ValueError("could not locate the text inside the tokenizer's special-token template")


def encode_head_tail(
    ids: list[int], prefix: list[int], suffix: list[int], max_length: int, tail_tokens: int
) -> list[int]:
    """Token ids (no special tokens) -> model input ids of at most `max_length` tokens.
    Long texts keep their start and their last `tail_tokens` tokens."""
    budget = max_length - len(prefix) - len(suffix)
    if len(ids) > budget:
        tail = min(tail_tokens, budget)
        ids = ids[: budget - tail] + ids[len(ids) - tail :]
    return prefix + list(ids) + suffix


def tokenize(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], max_length: int, tail_tokens: int
) -> list[list[int]]:
    prefix, suffix = special_tokens(tokenizer)
    raw = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [encode_head_tail(ids, prefix, suffix, max_length, tail_tokens) for ids in raw]


def class_index(example: Example, classes: list[str]) -> int:
    if example.label != ATTACK:
        return 0
    if example.family not in classes[1:]:
        raise ValueError(f"attack family {example.family!r} is not in classes {classes}")
    return classes.index(example.family)


def collate(
    seqs: list[list[int]], pad_id: int, device: str | torch.device
) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch (not to max_length)."""
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device)}


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    seqs: list[list[int]],
    batch_size: int,
) -> np.ndarray:
    """(n, classes) float32 logits. Batches are sorted by length to waste little padding."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    num_labels = model.config.num_labels
    assert num_labels is not None
    out = np.zeros((len(seqs), num_labels), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = collate([seqs[i] for i in idx], _pad_id(tokenizer), device)
        with _autocast(device):
            logits = model(**batch).logits
        out[idx] = logits.float().cpu().numpy()
    model.train(was_training)
    return out


def _pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad token")
    return int(tokenizer.pad_token_id)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_encoder(
    cfg: D1Config,
    train: list[Example],
    dev: list[Example],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    run_dir: Path,
    figure_path: Path,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, keep the best epoch by dev AUROC, calibrate on dev. Returns the calibration
    summary. `model` must already sit on the target device."""
    if cfg.classes[0] != "benign":
        raise ValueError("classes[0] must be 'benign'")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; bump d1.run_name in configs/detectors.yaml")
    _seed_everything(cfg.seed)
    device = next(model.parameters()).device

    train_seqs = tokenize(tokenizer, [e.text for e in train], cfg.max_length, cfg.tail_tokens)
    train_y = [class_index(e, cfg.classes) for e in train]
    dev_seqs = tokenize(tokenizer, [e.text for e in dev], cfg.max_length, cfg.tail_tokens)
    dev_y = np.array([1 if e.label == ATTACK else 0 for e in dev])

    steps_per_epoch = math.ceil(len(train) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M of "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"d1": cfg.model_dump(), "git_commit": git_commit(), **(extra_config or {})},
            indent=2,
        ),
        "utf-8",
    )

    history: list[dict[str, float]] = []
    best_auroc, best_dev_logits = -1.0, np.zeros(0)
    rng = random.Random(cfg.seed)
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        order = list(range(len(train_seqs)))
        rng.shuffle(order)
        micro_batches = [
            order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)
        ]
        losses: list[float] = []
        for step, idx in enumerate(micro_batches):
            batch = collate([train_seqs[i] for i in idx], _pad_id(tokenizer), device)
            labels = torch.tensor([train_y[i] for i in idx], device=device)
            with _autocast(device):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            (loss / cfg.grad_accum).backward()
            losses.append(loss.item())
            last = step == len(micro_batches) - 1
            if (step + 1) % cfg.grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                print(
                    f"  epoch {epoch} step {step + 1}/{len(micro_batches)} "
                    f"loss {np.mean(losses[-200:]):.4f}",
                    flush=True,
                )

        dev_logits = predict_logits(model, tokenizer, dev_seqs, cfg.eval_batch_size)
        p = sigmoid(attack_logit(dev_logits))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_auroc": auroc(dev_y, p),
            "dev_log_loss": log_loss(dev_y, p),
            "minutes": (time.perf_counter() - t0) / 60,
        }
        history.append(row)
        print(
            f"epoch {epoch}: train loss {row['train_loss']:.4f} | dev AUROC "
            f"{row['dev_auroc']:.4f} | {row['minutes']:.1f} min",
            flush=True,
        )
        if row["dev_auroc"] > best_auroc:
            best_auroc, best_dev_logits = row["dev_auroc"], dev_logits
            model.save_pretrained(run_dir / "model")
            tokenizer.save_pretrained(run_dir / "model")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), "utf-8")

    return calibrate_run(best_dev_logits, dev_y, run_dir, figure_path)


def calibrate_run(
    dev_logits: np.ndarray, dev_y: np.ndarray, run_dir: Path, figure_path: Path
) -> dict[str, Any]:
    margin = attack_logit(dev_logits)
    temperature = fit_temperature(margin, dev_y)
    before, after = sigmoid(margin), calibrated_score(margin, temperature)
    summary = {
        "temperature": temperature,
        "fitted_on": "dev",
        "dev_ece_before": ece(dev_y, before),
        "dev_ece_after": ece(dev_y, after),
        "dev_brier_before": brier(dev_y, before),
        "dev_brier_after": brier(dev_y, after),
    }
    (run_dir / "calibration.json").write_text(json.dumps(summary, indent=2), "utf-8")
    plot_reliability(dev_y, before, after, figure_path, "D1 calibration (dev)")
    return summary
