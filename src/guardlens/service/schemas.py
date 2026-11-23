"""Request / response models of the `/check` API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CheckRequest(BaseModel):
    text: str = Field(min_length=1, description="The text to screen")
    direction: str = Field(default="input", description="Only 'input' is supported")

    @field_validator("direction")
    @classmethod
    def _input_only(cls, value: str) -> str:
        if value != "input":
            raise ValueError(
                "only direction='input' is supported: the detectors were trained and evaluated "
                "on user inputs; output moderation is not implemented"
            )
        return value


class CheckResponse(BaseModel):
    allow: bool  # False = block
    score: float  # higher = more likely an attack
    threshold: float  # block when score >= threshold
    category: str  # predicted attack family, or "none"
    reason: str
    decided_by: Literal["d1", "d2"]
    mode: Literal["e2", "d1"]  # the mode that actually ran (after any fallback)
    fallback: str | None = None  # why the judge was skipped, if it should have been used
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    configured_mode: Literal["e2", "d1"]
    judge_available: bool
    judge_unavailable_reason: str | None = None
