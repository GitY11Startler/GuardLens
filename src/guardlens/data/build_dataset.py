"""`gl build-data`: load every source -> cap -> dedup -> split -> write versioned JSON.

Output in `data/processed/<version>/`:
  train.json, dev.json, test.json   lists of Example dicts
  manifest.json                     provenance + counts (no text)
  balance_report.csv                counts per split/source/family/category/label (no text)
"""

import csv
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from guardlens.data.config import DataConfig
from guardlens.data.dedup import DedupStats, deduplicate, exact_dedup
from guardlens.data.loaders import Loader, LoaderContext
from guardlens.data.schema import Example
from guardlens.data.splits import Splits, is_held_out, make_splits
from guardlens.run_info import git_commit


class SourceStats(BaseModel):
    hf_id: str
    license: str
    family: str | None
    loaded: int
    after_source_dedup: int
    after_cap: int


class Manifest(BaseModel):
    version: str
    created_at: str
    git_commit: str
    config_sha256: str
    seed: int
    held_out_families: list[str]
    hold_out_whole_source: bool
    sources: dict[str, SourceStats]
    dedup: DedupStats
    split_sizes: dict[str, int]
    held_out_in_test: int


def cap_per_label(examples: list[Example], cap: int, seed: int, source: str) -> list[Example]:
    """Seeded random sample of at most `cap` examples per label."""
    kept: list[Example] = []
    for label in (0, 1):
        group = sorted((e for e in examples if e.label == label), key=lambda e: e.id)
        if len(group) > cap:
            group = random.Random(f"{seed}:{source}:{label}").sample(group, cap)
        kept.extend(group)
    return sorted(kept, key=lambda e: e.id)


def balance_rows(splits: Splits, config: DataConfig) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str, str, str, int, bool]] = Counter()
    for split_name, part in (("train", splits.train), ("dev", splits.dev), ("test", splits.test)):
        for e in part:
            held = is_held_out(e, config)
            counts[(split_name, e.source, e.family or "", e.category, e.label, held)] += 1
    return [
        {
            "split": s,
            "source": src,
            "family": fam,
            "category": cat,
            "label": lab,
            "held_out": held,
            "count": n,
        }
        for (s, src, fam, cat, lab, held), n in sorted(counts.items())
    ]


def build(
    config: DataConfig,
    config_sha256: str,
    loaders: dict[str, Loader],
    ctx: LoaderContext,
    out_dir: Path,
    force: bool = False,
) -> Manifest:
    version_dir = out_dir / config.version
    if version_dir.exists() and not force:
        raise FileExistsError(
            f"{version_dir} already exists. Bump `version` in the config or pass --force."
        )

    pool: list[Example] = []
    source_stats: dict[str, SourceStats] = {}
    for name, src in config.sources.items():
        if not src.enabled:
            continue
        print(f"[load] {name} ({src.hf_id})")
        loaded = loaders[name](name, src, ctx)
        unique, _, _ = exact_dedup(loaded)
        capped = cap_per_label(unique, config.cap_for(name), config.seed, name)
        source_stats[name] = SourceStats(
            hf_id=src.hf_id,
            license=src.license,
            family=src.family,
            loaded=len(loaded),
            after_source_dedup=len(unique),
            after_cap=len(capped),
        )
        print(f"       loaded={len(loaded)} unique={len(unique)} kept={len(capped)}")
        pool.extend(capped)

    print(f"[dedup] {len(pool)} examples (exact + near-duplicate, all sources together)")
    deduped, dedup_stats = deduplicate(
        pool, config.dedup.threshold, config.dedup.shingle_size, config.dedup.num_perm
    )
    splits = make_splits(deduped, config)

    manifest = Manifest(
        version=config.version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        config_sha256=config_sha256,
        seed=config.seed,
        held_out_families=config.held_out_families,
        hold_out_whole_source=config.hold_out_whole_source,
        sources=source_stats,
        dedup=dedup_stats,
        split_sizes={"train": len(splits.train), "dev": len(splits.dev), "test": len(splits.test)},
        held_out_in_test=sum(is_held_out(e, config) for e in splits.test),
    )

    version_dir.mkdir(parents=True, exist_ok=True)
    for split_name, part in (("train", splits.train), ("dev", splits.dev), ("test", splits.test)):
        with (version_dir / f"{split_name}.json").open("w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in part], f, ensure_ascii=False)
    (version_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), "utf-8")

    rows = balance_rows(splits, config)
    with (version_dir / "balance_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {manifest.split_sizes} -> {version_dir}")
    return manifest


def load_split(processed_dir: Path, version: str, split: str) -> list[Example]:
    with (processed_dir / version / f"{split}.json").open(encoding="utf-8") as f:
        return [Example.model_validate(d) for d in json.load(f)]
