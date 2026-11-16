import json
from pathlib import Path

import pytest
from helpers import make_example

from guardlens.data.schema import ATTACK, BENIGN
from guardlens.eval.config import EnsembleConfig, LatencyConfig
from guardlens.eval.ensemble import (
    build_ensemble,
    combine,
    ensemble_cost,
    ensemble_latency,
)
from guardlens.eval.runner import Prediction, RunConfig, load_predictions, run_dir_for

MEAN = EnsembleConfig(kind="mean", components=["a", "b"])
CASCADE = EnsembleConfig(kind="cascade", components=["a", "b"], band=(0.1, 0.9))


def _pred(i: str, score: float, category: str = "fam_x", label: int = 1) -> Prediction:
    return Prediction(
        id=i,
        label=label,
        source="src",
        family="fam_x" if label else None,
        category="fam_x" if label else "benign",
        score=score,
        pred_category=category if score >= 0.5 else "none",
        reason="placeholder",
        batch_ms_per_example=1.0,
    )


def test_mean_averages_scores() -> None:
    out = combine(
        MEAN, {"a": [_pred("x", 0.2), _pred("y", 1.0)], "b": [_pred("y", 0.6), _pred("x", 0.4)]}
    )
    assert [p.id for p in out] == ["x", "y"]  # order of the first component
    assert [p.score for p in out] == pytest.approx([0.3, 0.8])
    assert out[0].pred_category == "none" and out[1].pred_category == "fam_x"


def test_cascade_uses_second_only_inside_band() -> None:
    a = [_pred("sure_attack", 0.95), _pred("sure_benign", 0.02), _pred("unsure", 0.5)]
    b = [_pred("sure_attack", 0.0), _pred("sure_benign", 1.0), _pred("unsure", 1.0, "other_attack")]
    out = {p.id: p for p in combine(CASCADE, {"a": a, "b": b})}
    assert out["sure_attack"].score == 0.95  # first is confident: second ignored
    assert out["sure_benign"].score == 0.02
    assert out["unsure"].score == pytest.approx(0.1 + 0.8 * 1.0)  # mapped into the band
    assert out["unsure"].pred_category == "other_attack"
    assert "unsure" in out["unsure"].reason


def test_cascade_never_reorders_confident_decisions() -> None:
    a = [_pred("hi", 0.91), _pred("mid", 0.89)]
    b = [_pred("hi", 0.0), _pred("mid", 1.0)]
    out = {p.id: p.score for p in combine(CASCADE, {"a": a, "b": b})}
    assert out["mid"] <= 0.9 < out["hi"]


def test_components_must_cover_the_same_examples() -> None:
    with pytest.raises(ValueError, match="same examples"):
        combine(MEAN, {"a": [_pred("x", 0.1)], "b": [_pred("z", 0.1)]})


def test_latency_and_cost() -> None:
    lat = {"a": [10.0, 20.0], "b": [1000.0, 2000.0]}
    first = {"x": 0.5, "y": 0.99}  # x inside the band, y outside
    assert ensemble_latency(MEAN, lat, ["x", "y"], first) == [1010.0, 2020.0]
    assert ensemble_latency(CASCADE, lat, ["x", "y"], first) == [1010.0, 20.0]
    assert ensemble_latency(MEAN, {"a": [1.0], "b": []}, ["x"], first) == []  # missing sample

    preds = [_pred("x", 0.5), _pred("y", 0.99), _pred("z", 0.3), _pred("w", 0.0)]
    assert ensemble_cost(MEAN, {"a": 0.0, "b": 0.2}, preds) == {"cost_per_1k_usd": 0.2}
    cost = ensemble_cost(CASCADE, {"a": 0.0, "b": 0.2}, preds)
    assert cost["share_sent_to_second"] == 0.5
    assert cost["cost_per_1k_usd"] == pytest.approx(0.1)


def _component_run(results: Path, name: str, scores: dict[str, float], lat: list[float]) -> None:
    run_dir = run_dir_for(results, name, "dev", "vtest")
    run_dir.mkdir(parents=True)
    cfg = RunConfig(
        detector={"name": name},
        split="dev",
        data_version="vtest",
        data_manifest_sha256="sha",
        git_commit="abc",
        eval_tag=None,
        created_at="now",
    )
    (run_dir / "run_config.json").write_text(cfg.model_dump_json(), "utf-8")
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, s in scores.items():
            f.write(_pred(i, s).model_dump_json() + "\n")
    (run_dir / "latency.json").write_text(json.dumps(lat), "utf-8")


def test_build_ensemble_writes_a_normal_run(tmp_path) -> None:
    examples = [make_example("alpha", ATTACK), make_example("kilo", BENIGN)]
    ids = [e.id for e in examples]
    _component_run(tmp_path, "a", {ids[0]: 0.5, ids[1]: 0.01}, [5.0, 5.0])
    _component_run(tmp_path, "b", {ids[0]: 0.9, ids[1]: 0.9}, [900.0, 900.0])
    (run_dir_for(tmp_path, "b", "dev", "vtest") / "cost.json").write_text(
        '{"cost_per_1k_usd": 0.2}', "utf-8"
    )

    run_dir = build_ensemble(
        "e2", CASCADE, "dev", "vtest", tmp_path, examples, LatencyConfig(sample=2, seed=0)
    )
    preds = {p.id: p.score for p in load_predictions(run_dir)}
    assert preds[ids[0]] == pytest.approx(0.1 + 0.8 * 0.9)
    assert preds[ids[1]] == 0.01
    assert json.loads((run_dir / "cost.json").read_text("utf-8"))["share_sent_to_second"] == 0.5
    assert sorted(json.loads((run_dir / "latency.json").read_text("utf-8"))) == [5.0, 905.0]
    cfg = json.loads((run_dir / "run_config.json").read_text("utf-8"))
    assert cfg["detector"]["component_configs"] == {"a": {"name": "a"}, "b": {"name": "b"}}


def test_build_ensemble_needs_component_runs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="gl eval"):
        build_ensemble("e1", MEAN, "dev", "vtest", tmp_path, [], LatencyConfig())
