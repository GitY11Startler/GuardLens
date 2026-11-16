from pathlib import Path

import numpy as np
import pytest
from fake_embedder import FakeEmbedder
from helpers import make_example

from guardlens.data.schema import ATTACK, BENIGN
from guardlens.detectors.d3_knn import KNNDetector
from guardlens.embeddings import CachedEmbedder

TRAIN = [
    make_example("alpha bravo charlie placeholder one", ATTACK, family="fam_x"),
    make_example("alpha bravo charlie placeholder two", ATTACK, family="fam_x"),
    make_example("alpha bravo delta placeholder three", ATTACK, family="fam_y"),
    make_example("kilo lima mike placeholder four", BENIGN),
    make_example("kilo lima mike placeholder five", BENIGN),
    make_example("kilo lima november placeholder six", BENIGN),
]


def test_neighbours_vote_for_attack_and_benign() -> None:
    d3 = KNNDetector(FakeEmbedder(), k=3).fit(TRAIN)
    attack_like, benign_like = d3.score_batch(["alpha bravo charlie", "kilo lima mike"])
    assert attack_like.score > 0.9
    assert attack_like.category == "fam_x"
    assert benign_like.score < 0.1
    assert benign_like.category == "none"
    assert "3/3" in attack_like.reason and "0/3" in benign_like.reason


def test_scores_in_range_and_single_matches_batch() -> None:
    d3 = KNNDetector(FakeEmbedder(), k=4).fit(TRAIN)
    texts = ["alpha kilo", "bravo lima", "zulu", "charlie mike placeholder"]
    batch = d3.score_batch(texts)
    assert all(0.0 <= o.score <= 1.0 for o in batch)
    assert batch == [d3.score(t) for t in texts]


def test_k_larger_than_train_set_is_clamped() -> None:
    d3 = KNNDetector(FakeEmbedder(), k=50).fit(TRAIN)
    assert "/6 nearest" in d3.score("alpha").reason


def test_unfitted_detector_raises() -> None:
    with pytest.raises(RuntimeError):
        KNNDetector(FakeEmbedder()).score("alpha")


def test_cache_only_embeds_new_texts_and_persists(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.embed(["one placeholder", "two placeholder", "one placeholder"])
    assert inner.calls == 2  # the repeated text is embedded once
    cached.embed(["two placeholder", "three placeholder"])
    assert inner.calls == 3

    reloaded_inner = FakeEmbedder()
    reloaded = CachedEmbedder(reloaded_inner, tmp_path)
    again = reloaded.embed(["one placeholder", "two placeholder", "one placeholder"])
    assert reloaded_inner.calls == 0
    np.testing.assert_allclose(first, again)


def test_cache_key_depends_on_model_and_prefix(tmp_path: Path) -> None:
    a = CachedEmbedder(FakeEmbedder(), tmp_path)
    other = FakeEmbedder()
    other.prefix = "query: "
    b = CachedEmbedder(other, tmp_path)
    assert a.key("same text") != b.key("same text")
