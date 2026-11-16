from helpers import make_example

from guardlens.data.dedup import deduplicate, normalize, shingles
from guardlens.data.schema import ATTACK, BENIGN

LONG_A = "the quick brown fox jumps over the lazy dog near the quiet river bank today"
LONG_B = "please summarize this article about gardening tips for small apartment balconies"


def test_normalize_collapses_case_space_and_unicode() -> None:
    assert normalize("  Hello \tWORLD \n") == "hello world"
    assert normalize("ｆｕｌｌｗｉｄｔｈ") == "fullwidth"


def test_shingles_short_text_is_single_shingle() -> None:
    assert shingles("abc", 5) == {"abc"}
    assert shingles("abcdef", 5) == {"abcde", "bcdef"}


def test_exact_duplicates_are_collapsed() -> None:
    examples = [
        make_example(LONG_A, source="s1"),
        make_example("  " + LONG_A.upper(), source="s2"),
        make_example(LONG_B),
    ]
    kept, stats = deduplicate(examples)
    assert len(kept) == 2
    assert stats.exact_removed == 1
    assert stats.output == 2


def test_near_duplicates_are_collapsed() -> None:
    examples = [make_example(LONG_A), make_example(LONG_A + "!"), make_example(LONG_B)]
    kept, stats = deduplicate(examples)
    assert len(kept) == 2
    assert stats.near_removed == 1


def test_different_texts_are_kept() -> None:
    examples = [make_example(LONG_A), make_example(LONG_B)]
    kept, stats = deduplicate(examples)
    assert len(kept) == 2
    assert stats.exact_removed == stats.near_removed == stats.conflict_dropped == 0


def test_conflicting_labels_drop_the_whole_group() -> None:
    examples = [
        make_example(LONG_A, label=BENIGN),
        make_example(LONG_A + ".", label=ATTACK, family="toy_family"),
        make_example(LONG_B),
    ]
    kept, stats = deduplicate(examples)
    assert [e.text for e in kept] == [LONG_B]
    assert stats.conflict_dropped == 2


def test_dedup_is_deterministic() -> None:
    examples = [make_example(LONG_A, source="s2"), make_example(LONG_A, source="s1")]
    kept_1, _ = deduplicate(examples)
    kept_2, _ = deduplicate(list(reversed(examples)))
    assert kept_1 == kept_2
