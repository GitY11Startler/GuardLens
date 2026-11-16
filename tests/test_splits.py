import pytest
from helpers import make_example

from guardlens.data.config import DataConfig, SourceConfig, SplitRatios
from guardlens.data.schema import ATTACK, BENIGN, Example
from guardlens.data.splits import make_splits


def toy_config(held_out: list[str], whole_source: bool = True, seed: int = 7) -> DataConfig:
    return DataConfig(
        version="test",
        seed=seed,
        default_cap=1000,
        held_out_families=held_out,
        hold_out_whole_source=whole_source,
        sources={
            "src_a": SourceConfig(hf_id="toy/a", license="test-only", family="fam_a"),
            "src_b": SourceConfig(hf_id="toy/b", license="test-only", family="fam_b"),
            "src_c": SourceConfig(hf_id="toy/c", license="test-only", family=None),
        },
    )


def toy_pool() -> list[Example]:
    pool = []
    for i in range(100):
        pool.append(make_example(f"placeholder attack a {i}", ATTACK, "src_a", "fam_a"))
        pool.append(make_example(f"placeholder benign a {i}", BENIGN, "src_a"))
        pool.append(make_example(f"placeholder attack b {i}", ATTACK, "src_b", "fam_b"))
        pool.append(make_example(f"placeholder benign b {i}", BENIGN, "src_b"))
        pool.append(make_example(f"placeholder benign c {i}", BENIGN, "src_c"))
    return pool


def test_held_out_family_only_in_test() -> None:
    splits = make_splits(toy_pool(), toy_config(held_out=["fam_b"]))
    assert all(e.family != "fam_b" for e in splits.train + splits.dev)
    assert sum(e.family == "fam_b" for e in splits.test) == 100


def test_whole_source_hold_out_includes_its_benign_examples() -> None:
    splits = make_splits(toy_pool(), toy_config(held_out=["fam_b"], whole_source=True))
    assert all(e.source != "src_b" for e in splits.train + splits.dev)


def test_family_only_hold_out_keeps_benign_from_that_source_in_training() -> None:
    splits = make_splits(toy_pool(), toy_config(held_out=["fam_b"], whole_source=False))
    assert any(e.source == "src_b" and e.label == BENIGN for e in splits.train)
    assert all(e.family != "fam_b" for e in splits.train + splits.dev)


def test_no_example_in_two_splits_and_none_lost() -> None:
    pool = toy_pool()
    splits = make_splits(pool, toy_config(held_out=["fam_b"]))
    ids = [e.id for e in splits.train + splits.dev + splits.test]
    assert len(ids) == len(set(ids)) == len(pool)


def test_same_seed_same_splits_different_seed_different_splits() -> None:
    a = make_splits(toy_pool(), toy_config(held_out=[], seed=1))
    b = make_splits(toy_pool(), toy_config(held_out=[], seed=1))
    c = make_splits(toy_pool(), toy_config(held_out=[], seed=2))
    assert a == b
    assert a.train != c.train


def test_ratios_respected_per_source_and_label() -> None:
    splits = make_splits(toy_pool(), toy_config(held_out=[]))
    for source, label in [("src_a", ATTACK), ("src_a", BENIGN), ("src_c", BENIGN)]:
        counts = [
            sum(e.source == source and e.label == label for e in part)
            for part in (splits.train, splits.dev, splits.test)
        ]
        assert counts == [70, 15, 15]


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.8, dev=0.15, test=0.15)
