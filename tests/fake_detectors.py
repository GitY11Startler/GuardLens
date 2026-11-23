"""A detector that always returns the same score (or raises), for service tests."""

from guardlens.detectors.base import Detector, DetectorOutput, Direction


class Fixed(Detector):
    def __init__(self, name: str, score: float, error: Exception | None = None) -> None:
        self.name = name
        self.value = score
        self.error = error
        self.calls = 0

    def score(self, text: str, direction: Direction = "input") -> DetectorOutput:
        self.calls += 1
        if self.error is not None:
            raise self.error
        category = "fam_x" if self.value >= 0.5 else "none"
        return DetectorOutput(score=self.value, category=category, reason=f"{self.name} says")
