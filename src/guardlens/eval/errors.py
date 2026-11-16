"""Seeded sample of one detector's errors for manual error analysis.

Cells: false negatives (missed attacks) and false positives (benign text flagged), each split
into seen / held-out, at the detector's main operating point (threshold set on dev).
The sample stores ids and metadata only, never text: the write-up describes errors by
category. Empty cells (e.g. held-out on dev) are simply skipped.
"""

import random
from typing import Any

from guardlens.eval.runner import Prediction


def error_sample(
    preds: list[Prediction],
    threshold: float,
    held_out_sources: frozenset[str],
    per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for kind in ("FN", "FP"):
        for slice_name in ("seen", "held_out"):
            pool = sorted(
                (
                    p
                    for p in preds
                    if (p.source in held_out_sources) == (slice_name == "held_out")
                    and (
                        (p.label == 1 and p.score < threshold)
                        if kind == "FN"
                        else (p.label == 0 and p.score >= threshold)
                    )
                ),
                key=lambda p: p.id,  # fixed order, so the seed alone decides the sample
            )
            for p in rng.sample(pool, min(per_cell, len(pool))):
                rows.append(
                    {
                        "kind": kind,
                        "slice": slice_name,
                        "cell_size": len(pool),
                        "id": p.id,
                        "source": p.source,
                        "category": p.category,
                        "score": p.score,
                        "pred_category": p.pred_category,
                        "error_type": "",  # filled in during the manual review
                    }
                )
    return rows
