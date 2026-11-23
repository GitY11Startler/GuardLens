"""Unified example schema shared by every data source."""

import hashlib
from typing import Literal

from pydantic import BaseModel

Label = Literal[0, 1]  # 1 = prompt-injection / jailbreak attempt, 0 = benign
BENIGN: Label = 0
ATTACK: Label = 1


class Example(BaseModel):
    id: str
    text: str
    label: Label
    category: str  # attacks: same as family; benign: "benign" / "hard_negative" / "multilingual"
    family: str | None  # attack family (None for benign)
    source: str  # our short source name, e.g. "deepset"
    license: str
    language: str = "en"
    hard_negative: bool = False


def make_id(source: str, text: str) -> str:
    """Stable id: same source + same text always gives the same id."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"
