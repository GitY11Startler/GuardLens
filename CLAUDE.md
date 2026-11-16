# CLAUDE.md — working rules for this repo

Read `PROJECT_SPEC.md` first. It is the source of truth for scope, architecture, milestones and evaluation protocol. If a request conflicts with the spec, say so before coding.

## Scope (non-negotiable)
- This is a **defensive** project: detectors that flag prompt-injection / jailbreak inputs and unsafe outputs, plus a benchmark of those detectors.
- Do **not** build anything that generates, mutates or optimizes attacks, and no "attacker agent". Published datasets are used only as labeled data for training and testing detectors.
- In reports, error analyses and the README, describe missed attacks **by category**, not by quoting the prompts verbatim.

## How we work
- Work **one milestone at a time** (M0 → M6 in the spec). Before starting a milestone, propose a short plan (files to create or change, tests) and wait for approval.
- Keep changes small and reviewable. After each step, run `uv run pytest -q` and `uv run ruff check .` and report results.
- Don't add a dependency, framework or service without asking first.
- Prefer simple, explicit Python over abstractions.
- Ask when something is ambiguous; don't guess on anything that affects evaluation results.

## Environment
- Python 3.12, env via `uv`. GPU: RTX 5070 8 GB (Blackwell, sm_120). PyTorch must be a CUDA ≥ 12.8 build.
- Only one heavy model on the GPU at a time (e.g. don't load the local LLM judge while training the encoder).
- Data lives in `data/` (git-ignored). Never commit datasets, model weights, API keys or raw predictions.
- Secrets come from `.env` (`ANTHROPIC_API_KEY`, etc.), loaded via pydantic-settings.

## Data rules
- Datasets are downloaded by script from their official sources, never vendored. Each example keeps `source` and `license` fields; drop any source whose license doesn't allow this use.
- Deduplicate (exact + near-duplicate) **before** splitting, so the same text never appears in both train and test.
- Splits are source/family-aware, with at least one attack family fully held out of training. Seeds fixed; splits saved as JSON and versioned.
- Benign hard negatives (security questions asked in good faith, fiction, role-play, multilingual text) are first-class data, not an afterthought.

## Evaluation integrity
- Thresholds, prompts and hyperparameters are tuned on **dev** only. Never tune on **test**.
- Test runs happen only from a tagged commit (`eval-v*`); every run writes its config + git hash to the results folder.
- All detectors implement the same interface (`score(text, direction) -> {score, category, reason}`) and are evaluated by the same runner on the same examples.
- Report curves (ROC, PR, FPR vs. catch rate), FPR at a fixed TPR, per-category and seen-vs-held-out results, latency and cost, all with 95% bootstrap CIs. Use paired tests for comparisons.

## Code conventions
- Type hints everywhere; pydantic models for configs, detector outputs and the API schema.
- Expensive calls (LLM judge, embeddings) are cached on disk, keyed by a hash of input + model + prompt version. Runners are resumable.
- Log tokens and cost for every API call; respect the spending cap in `configs/detectors.yaml`.
- Unit tests use a tiny synthetic labeled set (harmless placeholder strings). No network or GPU needed for `pytest` (mark GPU tests with `@pytest.mark.gpu`).

## Useful commands (to be implemented)
- `uv run gl build-data`
- `uv run gl train --detector d1`
- `uv run gl eval --detector all --split dev`
- `uv run gl serve`
- `uv run gl report`
