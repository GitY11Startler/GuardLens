"""Pydantic model for `configs/service.yaml`."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class Thresholds(BaseModel):
    d1: float  # used by the service
    e2: float  # used by the service
    d0: float = 0.000001  # the rest only by the demo
    d2: float = 0.5
    d3: float = 0.5
    e1: float = 0.5


class ServiceConfig(BaseModel):
    mode: Literal["e2", "d1"] = "e2"
    thresholds: Thresholds
    max_chars: int = 100_000
    d3_device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000


def load_service_config(path: Path) -> ServiceConfig:
    with path.open(encoding="utf-8") as f:
        return ServiceConfig.model_validate(yaml.safe_load(f))
