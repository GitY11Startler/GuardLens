"""Seen vs. held-out slices of a split.

Held out = every example from a source whose attack family is in `held_out_families`
(configs/data.yaml, with hold_out_whole_source: true): its attacks *and* its benign
look-alikes. These examples are only in test, so dev has no held-out slice.
"""

from guardlens.data.config import DataConfig
from guardlens.eval.runner import Prediction

SLICES = ("all", "seen", "held_out")


def held_out_sources(data_config: DataConfig) -> frozenset[str]:
    if not data_config.hold_out_whole_source:
        raise NotImplementedError("slices assume hold_out_whole_source: true")
    return frozenset(
        name
        for name, source in data_config.sources.items()
        if source.family in data_config.held_out_families
    )


def slices(preds: list[Prediction], held_out: frozenset[str]) -> dict[str, list[Prediction]]:
    """{"all": ...} plus "seen" / "held_out" when the split has held-out examples."""
    held = [p for p in preds if p.source in held_out]
    if not held:
        return {"all": preds}
    return {"all": preds, "seen": [p for p in preds if p.source not in held_out], "held_out": held}
