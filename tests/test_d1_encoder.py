import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tiny_encoder import tiny_model, tiny_tokenizer

from guardlens.detectors.base import NO_CATEGORY, Detector
from guardlens.detectors.d1_encoder import EncoderDetector
from guardlens.training.calibrate import sigmoid

CLASSES = ["benign", "fam_x", "fam_y"]
TEXTS = ["alpha bravo charlie", "kilo lima mike placeholder", "november"]


def _fake_run(run_dir: Path, temperature: float = 2.0) -> Path:
    torch.manual_seed(0)
    tiny_model(len(CLASSES)).save_pretrained(run_dir / "model")
    tiny_tokenizer().save_pretrained(run_dir / "model")
    cfg = {"model": "tiny", "run_name": "t", "classes": CLASSES, "max_length": 16, "tail_tokens": 4}
    (run_dir / "config.json").write_text(json.dumps({"d1": cfg}), "utf-8")
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": temperature}), "utf-8")
    return run_dir


def test_load_and_score_follows_interface(tmp_path) -> None:
    d1 = EncoderDetector.load(_fake_run(tmp_path), device="cpu")
    assert isinstance(d1, Detector)
    outs = d1.score_batch(TEXTS)
    assert len(outs) == 3
    for o in outs:
        assert 0.0 <= o.score <= 1.0
        assert o.category in {NO_CATEGORY, "fam_x", "fam_y"}
        assert "p(attack)" in o.reason
    assert d1.score(TEXTS[1]).score == pytest.approx(outs[1].score, abs=1e-5)
    assert d1.config()["temperature"] == 2.0
    assert d1.config()["base_model"] == "tiny"


def test_temperature_is_applied(tmp_path) -> None:
    run = _fake_run(tmp_path)
    t1 = EncoderDetector.load(run, "cpu")
    t1.temperature = 1.0
    t2 = EncoderDetector.load(run, "cpu")  # temperature 2.0 from calibration.json
    s1 = np.array([o.score for o in t1.score_batch(TEXTS)])
    s2 = np.array([o.score for o in t2.score_batch(TEXTS)])
    logit1 = np.log(s1 / (1 - s1))
    np.testing.assert_allclose(s2, sigmoid(logit1 / 2.0), atol=1e-5)


def test_category_only_when_score_is_high(tmp_path) -> None:
    d1 = EncoderDetector.load(_fake_run(tmp_path), "cpu")
    assert d1._output(0.2, "fam_x").category == NO_CATEGORY
    assert d1._output(0.8, "fam_y").category == "fam_y"


def test_load_refuses_unfinished_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="gl train"):
        EncoderDetector.load(tmp_path, "cpu")
