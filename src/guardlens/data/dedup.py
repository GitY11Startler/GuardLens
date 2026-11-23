"""Exact + near-duplicate removal, run on the whole pool before splitting.

Every duplicate group is reduced to one example (the one with the smallest id, so the
choice is deterministic). If a group mixes attack and benign labels, the whole group
is dropped: we can't tell which label is right.
"""

import re
import unicodedata

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel

from guardlens.data.schema import Example

_WHITESPACE = re.compile(r"\s+")


class DedupStats(BaseModel):
    input: int
    exact_removed: int
    near_removed: int
    conflict_dropped: int
    output: int


def normalize(text: str) -> str:
    """Canonical form used for comparison only (the stored text is untouched)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", text).strip()


def shingles(text: str, k: int) -> set[str]:
    """Character k-grams of an already-normalized text."""
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str, k: int, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    m.update_batch([s.encode("utf-8") for s in shingles(text, k)])
    return m


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def _reduce_groups(
    examples: list[Example], groups: dict[int, list[int]]
) -> tuple[list[Example], int, int]:
    """Keep one example per group. Returns (kept, removed_as_duplicate, dropped_for_conflict)."""
    kept: list[Example] = []
    removed = conflicts = 0
    for members in groups.values():
        labels = {examples[i].label for i in members}
        if len(labels) > 1:
            conflicts += len(members)
            continue
        best = min(members, key=lambda i: examples[i].id)
        kept.append(examples[best])
        removed += len(members) - 1
    kept.sort(key=lambda e: e.id)
    return kept, removed, conflicts


def exact_dedup(examples: list[Example]) -> tuple[list[Example], int, int]:
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(normalize(ex.text), []).append(i)
    return _reduce_groups(examples, dict(enumerate(groups.values())))


def near_dedup(
    examples: list[Example], threshold: float, shingle_size: int, num_perm: int
) -> tuple[list[Example], int, int]:
    hashes = [_minhash(normalize(ex.text), shingle_size, num_perm) for ex in examples]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(hashes):
        lsh.insert(i, m)

    uf = _UnionFind(len(examples))
    for i, m in enumerate(hashes):
        for j in lsh.query(m):
            # LSH returns candidates; confirm with the estimated Jaccard similarity.
            if j != i and m.jaccard(hashes[j]) >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(examples)):
        groups.setdefault(uf.find(i), []).append(i)
    return _reduce_groups(examples, groups)


def deduplicate(
    examples: list[Example], threshold: float = 0.8, shingle_size: int = 5, num_perm: int = 128
) -> tuple[list[Example], DedupStats]:
    after_exact, exact_removed, exact_conflicts = exact_dedup(examples)
    after_near, near_removed, near_conflicts = near_dedup(
        after_exact, threshold, shingle_size, num_perm
    )
    stats = DedupStats(
        input=len(examples),
        exact_removed=exact_removed,
        near_removed=near_removed,
        conflict_dropped=exact_conflicts + near_conflicts,
        output=len(after_near),
    )
    return after_near, stats
