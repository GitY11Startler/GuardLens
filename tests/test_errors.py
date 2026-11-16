from guardlens.eval.errors import error_sample
from guardlens.eval.runner import Prediction


def _pred(i: int, label: int, score: float, source: str = "src") -> Prediction:
    return Prediction(
        id=f"id{i:03d}",
        label=label,
        source=source,
        family="fam_x" if label else None,
        category="fam_x" if label else "benign",
        score=score,
        pred_category="none",
        reason="placeholder",
        batch_ms_per_example=1.0,
    )


PREDS = (
    [_pred(i, 1, 0.1) for i in range(10)]  # seen FN
    + [_pred(10 + i, 1, 0.9) for i in range(5)]  # seen TP
    + [_pred(20 + i, 0, 0.8) for i in range(4)]  # seen FP
    + [_pred(30 + i, 1, 0.2, "held") for i in range(6)]  # held-out FN
    + [_pred(40 + i, 0, 0.1, "held") for i in range(3)]  # held-out TN: no held-out FP
)


def test_cells_and_sizes() -> None:
    rows = error_sample(PREDS, 0.5, frozenset({"held"}), per_cell=3, seed=0)
    counts = {(r["kind"], r["slice"]): r["cell_size"] for r in rows}
    assert counts == {("FN", "seen"): 10, ("FN", "held_out"): 6, ("FP", "seen"): 4}
    assert sum(r["kind"] == "FN" and r["slice"] == "seen" for r in rows) == 3
    assert all(r["error_type"] == "" for r in rows)
    assert "text" not in rows[0]  # ids and metadata only


def test_sample_is_fixed_by_seed_and_not_by_input_order() -> None:
    a = error_sample(PREDS, 0.5, frozenset({"held"}), 3, seed=0)
    b = error_sample(list(reversed(PREDS)), 0.5, frozenset({"held"}), 3, seed=0)
    c = error_sample(PREDS, 0.5, frozenset({"held"}), 3, seed=1)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_small_cell_is_taken_whole() -> None:
    rows = error_sample(PREDS, 0.5, frozenset({"held"}), per_cell=25, seed=0)
    assert sum(r["kind"] == "FP" for r in rows) == 4
