import csv
import hashlib
import json
from pathlib import Path

import pytest
from helpers import make_example

from guardlens.data.build_dataset import build, cap_per_label, load_split
from guardlens.data.config import DataConfig, SourceConfig, load_data_config
from guardlens.data.loaders import LOADERS, LoaderContext
from guardlens.data.schema import ATTACK, BENIGN, Example
from guardlens.settings import PROJECT_ROOT

SHARED = "shared placeholder sentence that appears in several toy sources"


def placeholder(*parts: object) -> str:
    """Distinct harmless text (random-looking, so no two are near-duplicates)."""
    return "placeholder " + hashlib.sha256(repr(parts).encode()).hexdigest()


def fake_loader(n_attack: int, n_benign: int, family: str | None):
    def load(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
        out = [
            make_example(placeholder(name, "a", i), ATTACK, name, family or "")
            for i in range(n_attack)
        ]
        out += [make_example(placeholder(name, "b", i), BENIGN, name) for i in range(n_benign)]
        # An exact duplicate inside the source, and a text shared with other sources.
        out.append(make_example(placeholder(name, "b", 0), BENIGN, name))
        out.append(make_example(SHARED, BENIGN, name))
        return out

    return load


def toy_config() -> DataConfig:
    return DataConfig(
        version="vtest",
        seed=3,
        default_cap=30,
        held_out_families=["fam_held"],
        sources={
            "src_seen": SourceConfig(hf_id="toy/seen", license="mit", family="fam_seen"),
            "src_held": SourceConfig(hf_id="toy/held", license="mit", family="fam_held"),
            "src_benign": SourceConfig(hf_id="toy/benign", license="mit", family=None, cap=10),
        },
    )


TOY_LOADERS = {
    "src_seen": fake_loader(40, 20, "fam_seen"),
    "src_held": fake_loader(10, 10, "fam_held"),
    "src_benign": fake_loader(0, 25, None),
}


def run_toy_build(tmp_path: Path):
    ctx = LoaderContext(cache_dir=tmp_path / "raw", hf_token=None)
    return build(toy_config(), "sha-test", TOY_LOADERS, ctx, tmp_path / "processed")


def test_build_writes_versioned_outputs(tmp_path: Path) -> None:
    manifest = run_toy_build(tmp_path)
    out = tmp_path / "processed" / "vtest"
    for name in ("train.json", "dev.json", "test.json", "manifest.json", "balance_report.csv"):
        assert (out / name).exists()

    seen = manifest.sources["src_seen"]
    assert (seen.loaded, seen.after_source_dedup) == (62, 61)  # in-source duplicate removed
    assert seen.after_cap == 30 + 21  # attacks capped at 30, benign under the cap
    assert manifest.sources["src_benign"].after_cap == 10
    assert manifest.dedup.exact_removed >= 1  # SHARED survives in only one source
    assert manifest.dedup.near_removed == 0
    assert manifest.git_commit  # "no-git" until the repo exists

    test = load_split(tmp_path / "processed", "vtest", "test")
    assert manifest.held_out_in_test == sum(e.source == "src_held" for e in test) >= 20


def test_build_outputs_contain_no_overlap_and_no_held_out_in_train(tmp_path: Path) -> None:
    run_toy_build(tmp_path)
    processed = tmp_path / "processed"
    train, dev, test = (load_split(processed, "vtest", s) for s in ("train", "dev", "test"))
    ids = [e.id for e in train + dev + test]
    assert len(ids) == len(set(ids))
    assert all(e.source != "src_held" for e in train + dev)


def test_reports_contain_no_text(tmp_path: Path) -> None:
    run_toy_build(tmp_path)
    out = tmp_path / "processed" / "vtest"
    manifest_text = (out / "manifest.json").read_text("utf-8")
    report_text = (out / "balance_report.csv").read_text("utf-8")
    assert "placeholder" not in manifest_text + report_text
    assert SHARED not in manifest_text + report_text
    with (out / "balance_report.csv").open(encoding="utf-8") as f:
        assert sum(int(r["count"]) for r in csv.DictReader(f)) == sum(
            json.loads(manifest_text)["split_sizes"].values()
        )


def test_build_refuses_to_overwrite_a_version(tmp_path: Path) -> None:
    run_toy_build(tmp_path)
    with pytest.raises(FileExistsError):
        run_toy_build(tmp_path)


def test_cap_per_label_is_seeded() -> None:
    examples = [make_example(f"placeholder {i}") for i in range(50)]
    a = cap_per_label(examples, 10, seed=1, source="s")
    b = cap_per_label(list(reversed(examples)), 10, seed=1, source="s")
    assert a == b and len(a) == 10


def test_real_config_is_valid_and_every_source_has_a_loader() -> None:
    config = load_data_config(PROJECT_ROOT / "configs" / "data.yaml")
    assert set(config.sources) == set(LOADERS)
    assert config.held_out_families == ["adversarial_jailbreak"]
    for name, src in config.sources.items():
        assert src.license, name
