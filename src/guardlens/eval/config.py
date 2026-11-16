"""Pydantic models for `configs/eval.yaml` and `configs/detectors.yaml`."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from guardlens.judge_cost import Prices


class BootstrapConfig(BaseModel):
    resamples: int = 1000
    seed: int = 0


class LatencyConfig(BaseModel):
    sample: int = 200
    seed: int = 0
    warmup: int = 5


class EvalConfig(BaseModel):
    data_version: str
    target_tpr: float = 0.95
    # detector -> {operating point name: fixed threshold}
    extra_operating_points: dict[str, dict[str, float]] = {}
    bootstrap: BootstrapConfig = BootstrapConfig()
    batch_size: int = 256
    latency: LatencyConfig = LatencyConfig()


class D3Config(BaseModel):
    model: str
    prefix: str = ""
    k: int = 10
    batch_size: int = 64
    max_seq_length: int = 512
    device: str | None = None


class D1Config(BaseModel):
    model: str
    run_name: str  # checkpoint folder: data/models/d1/<run_name>/
    # Class 0 must be "benign"; the others are the attack families seen in training.
    classes: list[str]
    max_length: int = 512
    tail_tokens: int = 126  # long texts keep their start and their last `tail_tokens` tokens
    lr: float = 3e-5
    epochs: int = 3
    batch_size: int = 16  # per step
    grad_accum: int = 2  # effective batch = batch_size * grad_accum
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 13
    gradient_checkpointing: bool = False  # trades speed for VRAM
    # Keep the pretrained word-embedding table fixed. For multilingual models it holds most
    # of the parameters (mmBERT: ~197M of 307M) and doesn't fit in 8 GB when trained.
    freeze_embeddings: bool = False
    eval_batch_size: int = 64
    device: str = "cuda"


class D2Config(BaseModel):
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 2048  # thinking + the short JSON answer
    concurrency: int = 4  # API calls in flight at once
    max_retries: int = 5  # SDK retries on rate limits and server errors
    prices: Prices
    pilot_size: int = 200  # dev examples for prompt checks (balanced over categories)
    pilot_seed: int = 0


class DetectorsConfig(BaseModel):
    spending_cap_usd: float
    d1: D1Config
    d2: D2Config
    d3: D3Config


def load_eval_config(path: Path) -> EvalConfig:
    with path.open(encoding="utf-8") as f:
        return EvalConfig.model_validate(yaml.safe_load(f))


def load_detectors_config(path: Path) -> DetectorsConfig:
    with path.open(encoding="utf-8") as f:
        return DetectorsConfig.model_validate(yaml.safe_load(f))
