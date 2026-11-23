"""Pydantic model for `configs/data.yaml`."""

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class SourceConfig(BaseModel):
    hf_id: str  # Hugging Face dataset id
    license: str
    family: str | None  # attack family of this source's attacks (None if benign-only)
    cap: int | None = None  # max examples kept from this source (None = use default_cap)
    gated: bool = False  # needs HF_TOKEN + accepted terms on the Hub
    enabled: bool = True


class SplitRatios(BaseModel):
    train: float = 0.70
    dev: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitRatios":
        if abs(self.train + self.dev + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class DedupConfig(BaseModel):
    threshold: float = 0.8  # Jaccard similarity on character shingles
    shingle_size: int = 5
    num_perm: int = 128


class DataConfig(BaseModel):
    version: str
    seed: int
    default_cap: int
    split_ratios: SplitRatios = SplitRatios()
    # Families listed here never appear in train or dev; they go to test only.
    # Set to [] to put them back into normal training (e.g. adversarial_jailbreak).
    held_out_families: list[str] = []
    # If true, every example from a source whose family is held out (including its
    # benign look-alikes) goes to test, not just the attacks.
    hold_out_whole_source: bool = True
    dedup: DedupConfig = DedupConfig()
    sources: dict[str, SourceConfig]

    def cap_for(self, source: str) -> int:
        cap = self.sources[source].cap
        return self.default_cap if cap is None else cap


def load_data_config(path: Path) -> DataConfig:
    with path.open(encoding="utf-8") as f:
        return DataConfig.model_validate(yaml.safe_load(f))
