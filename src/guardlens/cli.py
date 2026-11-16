"""`gl` command-line entry point."""

import argparse
import json
import os
from pathlib import Path

# Commands that exist in the spec but are built in later milestones.
PLANNED_COMMANDS = {
    "serve": "M6",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gl", description="GuardLens command-line tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-env", help="Show Python/PyTorch/CUDA/GPU info and run a GPU test.")

    build_data = sub.add_parser("build-data", help="Download sources, dedup, split, write JSON.")
    build_data.add_argument("--config", type=Path, default=None, help="default: configs/data.yaml")
    build_data.add_argument("--force", action="store_true", help="overwrite an existing version")

    ev = sub.add_parser("eval", help="Score a split with one or all detectors.")
    ev.add_argument("--detector", default="all", help="d0, d1, d2, d3 or all")
    ev.add_argument("--split", choices=["dev", "test"], default="dev")

    tr = sub.add_parser("train", help="Fine-tune a learned detector on train, calibrate on dev.")
    tr.add_argument("--detector", choices=["d1"], default="d1")

    ens = sub.add_parser("ensemble", help="Build the ensembles in eval.yaml from existing runs.")
    ens.add_argument("--split", choices=["dev", "test"], default="dev")

    err = sub.add_parser("error-sample", help="Seeded sample of FN/FP ids for error analysis.")
    err.add_argument("--split", choices=["dev", "test"], default="dev")

    sub.add_parser(
        "judge-pilot",
        help="Score a small balanced dev sample with D2; print quality, cost and an estimate.",
    )

    rep = sub.add_parser("report", help="Summary CSVs + figures from existing runs.")
    rep.add_argument("--split", choices=["dev", "test"], default="dev")

    for name, milestone in PLANNED_COMMANDS.items():
        sub.add_parser(name, help=f"(not implemented yet: {milestone})")

    return parser


def use_project_hf_cache() -> None:
    """Keep Hugging Face downloads (datasets, models) inside data/ instead of ~/.cache.
    Must run before `datasets` / `huggingface_hub` / `sentence_transformers` are imported."""
    from guardlens.settings import get_settings

    os.environ.setdefault("HF_HOME", str(get_settings().data_dir / "raw" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def run_build_data(config_path: Path | None, force: bool) -> int:
    from guardlens.data.build_dataset import build
    from guardlens.data.config import load_data_config
    from guardlens.data.loaders import LOADERS, LoaderContext
    from guardlens.run_info import file_sha256
    from guardlens.settings import get_settings

    settings = get_settings()
    path = config_path or settings.configs_dir / "data.yaml"
    config = load_data_config(path)
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    ctx = LoaderContext(cache_dir=settings.data_dir / "raw" / "hf", hf_token=token)
    build(config, file_sha256(path), LOADERS, ctx, settings.data_dir / "processed", force)
    return 0


def run_eval(detector_arg: str, split: str) -> int:
    from guardlens.data.build_dataset import load_split
    from guardlens.detectors.registry import DETECTOR_NAMES, build_detector
    from guardlens.eval.config import load_detectors_config, load_eval_config
    from guardlens.eval.metrics import auroc, fpr_at_tpr
    from guardlens.eval.report import label_score_arrays
    from guardlens.eval.runner import (
        assert_split_allowed,
        load_predictions,
        measure_latency,
        run_detector,
        run_dir_for,
    )
    from guardlens.run_info import file_sha256
    from guardlens.settings import get_settings

    assert_split_allowed(split)  # fail before loading anything
    settings = get_settings()
    eval_cfg = load_eval_config(settings.configs_dir / "eval.yaml")
    det_cfg = load_detectors_config(settings.configs_dir / "detectors.yaml")
    processed = settings.data_dir / "processed"
    version = eval_cfg.data_version
    manifest_sha = file_sha256(processed / version / "manifest.json")
    examples = load_split(processed, version, split)
    train = load_split(processed, version, "train")

    names = DETECTOR_NAMES if detector_arg == "all" else [detector_arg]
    for name in names:
        detector = build_detector(name, det_cfg, train, settings.data_dir)
        run_dir = run_dir_for(settings.results_dir, name, split, version)
        run_detector(detector, examples, run_dir, split, version, manifest_sha, eval_cfg.batch_size)

        latency_path = run_dir / "latency.json"
        if not latency_path.exists():
            latency = measure_latency(detector, examples, eval_cfg.latency)
            latency_path.write_text(json.dumps(latency), "utf-8")

        cost_summary = getattr(detector, "cost_summary", None)
        if cost_summary is not None:  # paid detectors (D2): $ and tokens per API call
            (run_dir / "cost.json").write_text(json.dumps(cost_summary(), indent=2), "utf-8")

        y, s = label_score_arrays(load_predictions(run_dir))
        print(
            f"[{name}] {split}: AUROC {auroc(y, s):.3f} | "
            f"FPR@{eval_cfg.target_tpr:.0%} TPR {fpr_at_tpr(y, s, eval_cfg.target_tpr):.3f} "
            f"-> {run_dir}"
        )
    return 0


def run_train(detector: str) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from guardlens.data.build_dataset import load_split
    from guardlens.detectors.registry import d1_run_dir
    from guardlens.eval.config import load_detectors_config, load_eval_config
    from guardlens.run_info import file_sha256
    from guardlens.settings import get_settings
    from guardlens.training.train_encoder import train_encoder

    settings = get_settings()
    cfg = load_detectors_config(settings.configs_dir / "detectors.yaml").d1
    version = load_eval_config(settings.configs_dir / "eval.yaml").data_version
    processed = settings.data_dir / "processed"
    run_dir = d1_run_dir(settings.data_dir, cfg.run_name)
    if run_dir.exists():
        print(f"{run_dir} already exists; bump d1.run_name in configs/detectors.yaml")
        return 1
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("d1.device is cuda but no CUDA GPU is available")
        return 1

    if cfg.device == "cuda":
        # Fail with a clear out-of-memory error instead of letting Windows spill GPU memory
        # into system RAM (which makes training crawl without any error).
        torch.cuda.set_per_process_memory_fraction(0.90)

    train = load_split(processed, version, "train")
    dev = load_split(processed, version, "dev")  # used for epoch choice + calibration only
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model,
        num_labels=len(cfg.classes),
        id2label=dict(enumerate(cfg.classes)),
        label2id={c: i for i, c in enumerate(cfg.classes)},
    ).to(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    print(f"[{detector}] training {cfg.model} on {version} train ({len(train)} examples)")
    summary = train_encoder(
        cfg,
        train,
        dev,
        model,
        tokenizer,
        run_dir,
        settings.results_dir / "figures" / "d1_calibration_dev.png",
        extra_config={
            "data_version": version,
            "manifest_sha256": file_sha256(processed / version / "manifest.json"),
        },
    )
    print(
        f"[{detector}] T = {summary['temperature']:.3f} | dev ECE "
        f"{summary['dev_ece_before']:.4f} -> {summary['dev_ece_after']:.4f} -> {run_dir}"
    )
    return 0


def run_judge_pilot() -> int:
    import time

    import numpy as np

    from guardlens.data.build_dataset import load_split
    from guardlens.detectors.d2_prompt import PROMPT_VERSION
    from guardlens.detectors.registry import build_judge
    from guardlens.eval.config import load_detectors_config, load_eval_config
    from guardlens.eval.metrics import auroc
    from guardlens.eval.runner import pilot_sample
    from guardlens.settings import get_settings

    settings = get_settings()
    det_cfg = load_detectors_config(settings.configs_dir / "detectors.yaml")
    c = det_cfg.d2
    version = load_eval_config(settings.configs_dir / "eval.yaml").data_version
    dev = load_split(settings.data_dir / "processed", version, "dev")
    pilot = pilot_sample(dev, c.pilot_size, c.pilot_seed)
    judge = build_judge(det_cfg, settings.data_dir)

    spent_before = judge.ledger.total_usd
    print(f"[d2 pilot] {c.model}, effort={c.effort}, {PROMPT_VERSION}: {len(pilot)} dev texts")
    t0 = time.perf_counter()
    outs = judge.score_batch([e.text for e in pilot])
    minutes = (time.perf_counter() - t0) / 60
    spent = judge.ledger.total_usd - spent_before

    y = np.array([e.label for e in pilot])
    s = np.array([o.score for o in outs])
    rows = []
    for cat in sorted({e.category for e in pilot}):
        idx = [i for i, e in enumerate(pilot) if e.category == cat]
        rows.append(
            {
                "category": cat,
                "n": len(idx),
                "flagged_at_0.5": int(sum(s[i] >= 0.5 for i in idx)),
                "mean_score": float(np.mean(s[idx])),
            }
        )
    costs = judge.cost_summary()
    summary = {
        "model": c.model,
        "effort": c.effort,
        "prompt_version": PROMPT_VERSION,
        "n": len(pilot),
        "auroc": auroc(y, s),
        "refusals": sum(o.category == "judge_refusal" for o in outs),
        "per_category": rows,
        "spent_this_run_usd": spent,
        "minutes": minutes,
        **costs,
        "estimated_full_dev_usd": costs["cost_per_1k_usd"] * len(dev) / 1000,
        "total_spent_usd": judge.ledger.total_usd,
        "spending_cap_usd": det_cfg.spending_cap_usd,
    }
    out_dir = settings.results_dir / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d2_pilot_{PROMPT_VERSION}_{c.effort}.json"
    out_path.write_text(json.dumps(summary, indent=2), "utf-8")

    for r in rows:
        print(
            f"  {r['category']:<20} n={r['n']:<3} flagged {r['flagged_at_0.5']:<3} "
            f"mean score {r['mean_score']:.2f}"
        )
    print(
        f"[d2 pilot] AUROC {summary['auroc']:.3f} | refusals {summary['refusals']} | "
        f"{minutes:.1f} min | spent ${spent:.4f} this run "
        f"(${judge.ledger.total_usd:.4f} of ${det_cfg.spending_cap_usd:.2f} total)"
    )
    print(
        f"[d2 pilot] per call: {costs['mean_input_tokens']:.0f} in / "
        f"{costs['mean_output_tokens']:.0f} out tokens | full dev run ~ "
        f"${summary['estimated_full_dev_usd']:.2f} -> {out_path}"
    )
    return 0


def run_ensembles(split: str) -> int:
    from guardlens.data.build_dataset import load_split
    from guardlens.eval.config import load_eval_config
    from guardlens.eval.ensemble import build_ensemble
    from guardlens.settings import get_settings

    settings = get_settings()
    config = load_eval_config(settings.configs_dir / "eval.yaml")
    examples = load_split(settings.data_dir / "processed", config.data_version, split)
    for name, cfg in config.ensembles.items():
        run_dir = build_ensemble(
            name, cfg, split, config.data_version, settings.results_dir, examples, config.latency
        )
        cost = json.loads((run_dir / "cost.json").read_text("utf-8"))
        extra = (
            f" | {cost['share_sent_to_second']:.1%} of texts sent to {cfg.components[1]}"
            if "share_sent_to_second" in cost
            else ""
        )
        print(f"[{name}] {cfg.kind} of {', '.join(cfg.components)}{extra} -> {run_dir}")
    return 0


def run_error_sample(split: str) -> int:
    import csv

    from guardlens.data.config import load_data_config
    from guardlens.eval.config import load_eval_config
    from guardlens.eval.errors import error_sample
    from guardlens.eval.report import operating_points
    from guardlens.eval.runner import load_predictions, run_dir_for
    from guardlens.eval.slices import held_out_sources
    from guardlens.settings import get_settings

    settings = get_settings()
    config = load_eval_config(settings.configs_dir / "eval.yaml")
    c = config.error_sample
    held = held_out_sources(load_data_config(settings.configs_dir / "data.yaml"))
    dev_preds = load_predictions(
        run_dir_for(settings.results_dir, c.detector, "dev", config.data_version)
    )
    points = operating_points(c.detector, dev_preds, config)
    point = config.main_operating_point.get(c.detector, f"tpr{int(config.target_tpr * 100)}_on_dev")
    preds = load_predictions(
        run_dir_for(settings.results_dir, c.detector, split, config.data_version)
    )
    rows = error_sample(preds, points[point], held, c.per_cell, c.seed)

    out = settings.results_dir / "error_analysis" / f"{c.detector}_{split}_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    cells = sorted({(r["kind"], r["slice"], r["cell_size"]) for r in rows})
    for kind, slice_name, size in cells:
        n = sum(r["kind"] == kind and r["slice"] == slice_name for r in rows)
        print(f"[{c.detector}] {kind} {slice_name}: sampled {n} of {size}")
    print(f"[{c.detector}] {point} (threshold {points[point]:.4f}) -> {out}")
    return 0


def run_report(split: str) -> int:
    from guardlens.data.config import load_data_config
    from guardlens.detectors.registry import DETECTOR_NAMES
    from guardlens.eval.config import load_eval_config
    from guardlens.eval.report import build_report
    from guardlens.eval.slices import held_out_sources
    from guardlens.settings import get_settings

    settings = get_settings()
    config = load_eval_config(settings.configs_dir / "eval.yaml")
    held = held_out_sources(load_data_config(settings.configs_dir / "data.yaml"))
    names = DETECTOR_NAMES + list(config.ensembles)
    rows = build_report(names, split, config, settings.results_dir, held)
    for r in rows:
        print(
            f"{r['detector']} @ {r['operating_point']} [{r['slice']}]: "
            f"AUROC {r['auroc']:.3f} [{r['auroc_low']:.3f}, {r['auroc_high']:.3f}]"
            f" | catch {r['recall']:.3f} | FPR {r['fpr']:.3f} | F1 {r['f1']:.3f}"
            f" | latency p50 {r['latency_p50_ms']:.2f} ms"
        )
    print(f"-> {settings.results_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check-env":
        from guardlens.env_check import check_env

        return check_env()

    use_project_hf_cache()
    if args.command == "build-data":
        return run_build_data(args.config, args.force)
    if args.command == "eval":
        return run_eval(args.detector, args.split)
    if args.command == "report":
        return run_report(args.split)
    if args.command == "train":
        return run_train(args.detector)
    if args.command == "judge-pilot":
        return run_judge_pilot()
    if args.command == "ensemble":
        return run_ensembles(args.split)
    if args.command == "error-sample":
        return run_error_sample(args.split)

    milestone = PLANNED_COMMANDS[args.command]
    print(f"`gl {args.command}` is not implemented yet (planned for {milestone}).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
