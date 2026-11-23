"""D1: fine-tuned encoder classifier (see guardlens.training.train_encoder).

Score = calibrated probability that the text is an attack:
sigmoid(attack logit / T), with T fitted on dev. Category = the most likely attack family
when the score is at least 0.5, else "none".
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from guardlens.detectors.base import NO_CATEGORY, Detector, DetectorOutput, Direction
from guardlens.training.calibrate import attack_logit, calibrated_score
from guardlens.training.train_encoder import predict_logits, tokenize


class EncoderDetector(Detector):
    name = "d1"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        classes: list[str],
        temperature: float,
        max_length: int,
        tail_tokens: int,
        batch_size: int = 64,
        run_name: str = "",
        base_model: str = "",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.classes = classes
        self.temperature = temperature
        self.max_length = max_length
        self.tail_tokens = tail_tokens
        self.batch_size = batch_size
        self.run_name = run_name
        self.base_model = base_model

    @classmethod
    def load(cls, run_dir: Path, device: str, batch_size: int = 64) -> "EncoderDetector":
        """Load a finished training run (best checkpoint + dev-fitted temperature)."""
        if not (run_dir / "calibration.json").exists():
            raise FileNotFoundError(f"no finished D1 run in {run_dir}; run `gl train` first")
        cfg = json.loads((run_dir / "config.json").read_text("utf-8"))["d1"]
        calibration = json.loads((run_dir / "calibration.json").read_text("utf-8"))
        model = AutoModelForSequenceClassification.from_pretrained(run_dir / "model").to(device)
        return cls(
            model,
            AutoTokenizer.from_pretrained(run_dir / "model"),
            classes=cfg["classes"],
            temperature=calibration["temperature"],
            max_length=cfg["max_length"],
            tail_tokens=cfg["tail_tokens"],
            batch_size=batch_size,
            run_name=cfg["run_name"],
            base_model=cfg["model"],
        )

    def score(self, text: str, direction: Direction = "input") -> DetectorOutput:
        return self.score_batch([text], direction)[0]

    def score_batch(self, texts: list[str], direction: Direction = "input") -> list[DetectorOutput]:
        seqs = tokenize(self.tokenizer, texts, self.max_length, self.tail_tokens)
        logits = predict_logits(self.model, self.tokenizer, seqs, self.batch_size)
        scores = calibrated_score(attack_logit(logits), self.temperature)
        families = np.argmax(logits[:, 1:], axis=1) + 1
        return [
            self._output(float(s), self.classes[f]) for s, f in zip(scores, families, strict=True)
        ]

    def _output(self, score: float, family: str) -> DetectorOutput:
        category = family if score >= 0.5 else NO_CATEGORY
        reason = f"encoder: p(attack) = {score:.2f}"
        if score >= 0.5:
            reason += f"; most likely attack family: {family}"
        return DetectorOutput(score=min(max(score, 0.0), 1.0), category=category, reason=reason)

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_name": self.run_name,
            "base_model": self.base_model,
            "classes": self.classes,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "tail_tokens": self.tail_tokens,
        }
