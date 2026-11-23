"""Source/family-aware train/dev/test splits.

1. Held-out examples (family in `held_out_families`, or their whole source if
   `hold_out_whole_source`) go to test only.
2. Everything else is split per (source, label) stratum with a seeded shuffle, so every
   source and both labels appear in train/dev/test in the configured proportions.

Dedup must already have run: then no text (or near-copy) can land in two splits.
"""

import random

from pydantic import BaseModel

from guardlens.data.config import DataConfig
from guardlens.data.schema import Example


class Splits(BaseModel):
    train: list[Example]
    dev: list[Example]
    test: list[Example]


def held_out_sources(config: DataConfig) -> set[str]:
    return {
        name
        for name, src in config.sources.items()
        if src.family is not None and src.family in config.held_out_families
    }


def is_held_out(example: Example, config: DataConfig) -> bool:
    if example.family in config.held_out_families:
        return True
    return config.hold_out_whole_source and example.source in held_out_sources(config)


def make_splits(examples: list[Example], config: DataConfig) -> Splits:
    rng = random.Random(config.seed)
    ratios = config.split_ratios
    splits = Splits(train=[], dev=[], test=[])

    strata: dict[tuple[str, int], list[Example]] = {}
    for ex in sorted(examples, key=lambda e: e.id):
        if is_held_out(ex, config):
            splits.test.append(ex)
        else:
            strata.setdefault((ex.source, ex.label), []).append(ex)

    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        n_train = round(len(group) * ratios.train)
        n_dev = round(len(group) * ratios.dev)
        splits.train.extend(group[:n_train])
        splits.dev.extend(group[n_train : n_train + n_dev])
        splits.test.extend(group[n_train + n_dev :])

    for part in (splits.train, splits.dev, splits.test):
        part.sort(key=lambda e: e.id)
    return splits
