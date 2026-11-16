# GuardLens — session handoff

> Updated 2026-11-14 at the end of session 2 (M3 + M4 done). Read this, `CLAUDE.md` and `PROJECT_SPEC.md`
> before doing anything. Update this file at the end of every session.

## 1. Where we are

| Milestone | Status |
|---|---|
| M0 Setup | ✅ done |
| M1 Data layer | ✅ done (data `v2` is the one in use) |
| M2 Baselines D0 + D3 on dev | ✅ done (incl. extra D0 operating point) |
| M3 Encoder D1 | ✅ done (mmBERT, frozen embeddings, run `mmbert-v1`) |
| M4 LLM judge D2 | ✅ done (Haiku 5.5, effort low, `d2-prompt-v3`; $0.98 of $5 spent) |
| **M5 Full benchmark on test** | **next: propose the plan and wait for approval. Needs git first.** |
| M6 serve/demo/README | not started |

Checks at handoff: `uv run pytest -q` → 100 passed (3 GPU tests deselected);
`uv run pytest -q -m gpu` → 3 passed; `uv run ruff check .` and `uv run mypy src tests` clean.

## 2. How the user wants to work

- **One milestone at a time.** Propose a short plan (files, tests, decisions with a
  recommendation), **wait for approval**, then build in small steps, running pytest + ruff
  (and mypy) after each step and reporting results.
- **Ask before adding any dependency.** Explain trade-offs simply; the user is learning and
  asks for plain-language explanations when something is unclear.
- **Defensive scope only** (see CLAUDE.md): no attack generation; describe missed attacks by
  category, never quote prompts; tests use harmless placeholder strings.
- **Never tune on test.** Test runs only from a git commit tagged `eval-v*`.
- When a long job runs (training, spikes, API runs), write output to a log file and tell the
  user how to follow it live (`Get-Content <log> -Wait`, `nvidia-smi -l 2`,
  `Get-Content data\cache\judge\spend.jsonl -Wait`); they asked for this every time.
- Ask before any paid API run beyond what the approved plan covers (give the $ estimate).
- Check a model's real parameter count and memory before recommending it (M3 lesson:
  mmBERT's 256k-word vocabulary made it 307M parameters, not "base-size").
- The user sometimes edits files themselves (e.g. added timing lines to `env_check.py`);
  keep their edits.

## 3. Milestone archive (do this for every milestone)

Folder: `C:\Users\yas1z\Downloads\GuardLens Milestones\M<n>\` (sibling of the workspace).
The user rewrites the project by hand from it to learn.

1. **When proposing the plan:** write it to `M<n>/M<n>.md` (create the folder). Don't
   overwrite an `M<n>.md` the user wrote.
2. **Before changing any code:** snapshot the project as the diff baseline, in the session's
   scratchpad (the scratchpad is per-session; the old baselines are gone):
   `tar --exclude=./.venv --exclude=./data --exclude=./.env --exclude=./results/runs --exclude='__pycache__' --exclude='.*_cache' -cf - . | tar -xf - -C <scratchpad>/baseline_M<n-1>`
   (anchor excludes with `./` — a bare `--exclude=data` also drops `src/guardlens/data/`).
3. **At the end:** copy every new/changed file into `M<n>/` at its project path, write
   `M<n>/changes.diff` (`diff -ruN` vs. the baseline, excluding `uv.lock`, `*.csv`, `*.png`),
   and `M<n>/M<n>_result.md`: decisions, pipeline, **suggested order to rewrite the files by
   hand**, how to run, results, caveats. Re-sync the archive if fixes happen afterwards.
4. Never archive `.env` or raw predictions; grep the archive for `hf_` tokens before finishing.

## 4. Environment facts

- **Native Windows 11** (not WSL2, overriding the spec). Shells: PowerShell and Git Bash.
- **Python 3.12** (user's install, 3.12.8), venv at `.venv/` managed by **uv 0.13.7**
  (installed with winget). In Bash, uv is not on PATH by default:
  `export PATH="$PATH:$(cygpath "$LOCALAPPDATA")/Microsoft/WinGet/Links"`.
  In PowerShell: `$env:Path = [Environment]::GetEnvironmentVariable('Path','User') + ';' + [Environment]::GetEnvironmentVariable('Path','Machine')`.
- **GPU**: RTX 5070 Laptop, 8 GB, sm_120. **torch 2.15.0+cu132** and torchvision from the
  PyTorch cu132 index (pinned via `[tool.uv.sources]` in `pyproject.toml`; user's choice).
- **No git repo yet** (user will add it later). `run_info.git_commit()` returns `"no-git"`;
  the runner refuses `--split test` until a commit is tagged `eval-v*`. Git must exist before M5.
- `.env` has a working `HF_TOKEN` and `ANTHROPIC_API_KEY` (never print, read back or archive
  them). **Spending cap $5 total** (`spending_cap_usd`); **$0.98 spent** after M4, ledger in
  `data/cache/judge/spend.jsonl`. If the cap is reached, the user wants to switch D2 to a
  local Ollama model (installed at `%LOCALAPPDATA%\Programs\Ollama`).
- Windows console is cp1252: don't print non-ASCII symbols (e.g. ≈) from CLI code when
  output is redirected, or it crashes with UnicodeEncodeError.
- **All Hugging Face downloads live in `data/raw/hf/`** (`HF_HOME` set by
  `cli.use_project_hf_cache()` and `tests/conftest.py`), never in `~/.cache`.
- Dependencies (all approved): torch, torchvision, pydantic, pydantic-settings, datasets,
  pyyaml, datasketch, sentence-transformers, scikit-learn, matplotlib, numpy, transformers,
  anthropic (2.8.0, built on `httpx2`);
  dev: pytest, ruff, mypy, types-pyyaml. Build backend: hatchling.
- `data/` ≈ 7 GB: `raw/hf` (datasets + e5 + mmBERT), `processed/v1` (kept for reference,
  superseded), `processed/v2` (in use), `cache/embeddings` (D3 vectors),
  `models/d1/mmbert-v1/` (D1 checkpoint ~1.2 GB + config/history/calibration JSON).
- **VRAM reality**: ~7.2 GB usable (Windows keeps ~1.1 GB). When PyTorch overflows, Windows
  silently spills into system RAM and jobs crawl with no error, so cap with
  `torch.cuda.set_per_process_memory_fraction(0.90)` (`gl train` does this).
- Stopping a stuck background process with `Stop-Process` is blocked by the permission
  classifier; stop it with the TaskStop tool (background task id) instead.

## 5. Key decisions so far (all approved by the user)

**Data (M1)**
- Attack = prompt injection or jailbreak attempt. **Plain harmful requests without
  manipulation are excluded** (JBB-Behaviors, XSTest "unsafe", WildJailbreak "vanilla").
- 12 sources (`configs/data.yaml`): deepset, gandalf, mosscap, spml (injection);
  in_the_wild, jackhhao, wildjailbreak (jailbreak); xstest (safe only), orbench_hard,
  orbench_80k, dolly, aya (ar/fr/es/de) (benign / hard negatives / multilingual).
  Excluded: xTRam1/safe-guard (no license).
- Families: `direct_injection`, `chatbot_injection`, `jailbreak_template`,
  `adversarial_jailbreak`. **Held out: `adversarial_jailbreak` (WildJailbreak), test only**,
  whole source including its benign look-alikes (`hold_out_whole_source: true`).
  Reversible via `held_out_families: []`.
- Caps per (source, label): 5,000 default; orbench_80k 2,000; **wildjailbreak 1,500 (v2)**.
- Dedup before split: exact (NFKC + casefold + whitespace) + MinHash near-dup
  (char 5-grams, Jaccard ≥ 0.8); groups with conflicting labels dropped.
- Splits 70/15/15 per (source, label), seed 13. Versions bumped **by hand** in
  `configs/data.yaml`; build refuses to overwrite an existing version without `--force`.
- WildJailbreak TSV needs `delimiter="\t", keep_default_na=False`.
- **v2 sizes**: train 24,328 · dev 5,214 (1,844 attacks) · test 8,212 (3,000 held out).

**Evaluation (M2)**
- Shared interface `Detector.score(text, direction) -> DetectorOutput{score, category, reason}`.
- D0: regex rules (`d0-rules-v1`) in `detectors/d0_heuristic.py`, noisy-OR of rule weights.
- D3: `intfloat/multilingual-e5-base` with `"query: "` prefix, k = 10, exact numpy cosine
  search over **train** (attacks + benign), similarity-weighted attack share. Embeddings
  cached by sha256(cache version + model + prefix + text).
- Operating point: threshold catching **95 % of dev attacks** (`tpr95_on_dev`), reused
  unchanged on test. Extra fixed points in `configs/eval.yaml` → `extra_operating_points`
  (D0: `any_rule_fires` = 1e-6). Every report row has an `operating_point` column.
- Bootstrap 95 % CIs (1,000 resamples, seed 0), paired bootstrap for comparisons, Wilson
  intervals for per-group rates. Latency: 200 dev texts one at a time, cache bypassed.
- Runs in `results/runs/<det>_<split>_<version>/` (resumable; refuses a changed config).

**Encoder D1 (M3)**
- `jhu-clsp/mmBERT-base` (22 layers, 307.5M params). **Word-embedding table frozen**
  (197M params; 110.9M trainable): full training needed ~4.9 GB fixed memory and didn't fit.
- 4 classes: benign (benign + hard_negative + multilingual merged) + the 3 seen families.
  Score = sigmoid(attack logit / T), attack logit = logsumexp(attack) − benign. Category =
  argmax attack family if score ≥ 0.5, else `none`.
- 512 tokens, head + tail truncation (first 384 + last 126). Transformers 5 has no
  `build_inputs_with_special_tokens`; `special_tokens()` probes the tokenizer instead.
- Plain PyTorch loop: AdamW lr 3e-5, warmup 6 %, wd 0.01, bf16 autocast, batch 8 × accum 4,
  3 epochs, best epoch by dev AUROC (epoch 2), seed 13. ~0.33 GB activations per 512-token text.
- Temperature scaling fit on dev (grid search), T = 2.08. Checkpoints in
  `data/models/d1/<run_name>/`; bump `run_name` to retrain (never overwrites).
- `build_detector(name, config, train, data_dir)` now takes `data_dir` (was `cache_dir`).

**LLM judge D2 (M4)**
- `claude-haiku-5-5`, `output_config={"effort": "low", "format": json_schema}`, max_tokens
  2048, 4 parallel calls, SDK max_retries 5. Answer: attack_probability 0–100, category
  (benign / 3 families / other_attack), reason (no quoting). Score = probability / 100.
- Rubric `d2-prompt-v3` in `detectors/d2_prompt.py` (bump `PROMPT_VERSION` on any edit: it
  is in the cache key). Text between `<text_to_classify>` markers, declared untrusted.
- Refusal → score 1.0, category `judge_refusal` (3 on dev). `max_tokens` stop → JudgeError.
- Cache: `data/cache/judge/answers/<2 hex>/<sha256>.json`. Ledger checks worst-case $ before
  every call. `gl eval` writes `cost.json` in the run folder → report `cost_per_1k_usd`.
- `gl judge-pilot`: 200 balanced dev texts (33 per category, seed 0) → `results/pilot/`.
- Extra fixed operating point `judge_verdict` = 0.5 in `configs/eval.yaml` (the 95 %-catch
  threshold is degenerate for D2: 0.02).
- `Detector.without_cache()` hook (base no-op; D2 and D3 override); runner uses it for latency.

## 6. Dev results so far (data v2)

| | D0 @ any_rule_fires | D1 @ tpr95 | D2 @ judge_verdict | D3 @ tpr95 |
|---|---|---|---|---|
| AUROC [95 % CI] | 0.669 [0.657, 0.681] | 0.996 [0.995, 0.997] | 0.875 [0.866, 0.884] | 0.989 [0.987, 0.992] |
| Catch rate / FPR | 0.412 / 0.088 | 0.950 / 0.016 | 0.655 / 0.108 | 0.950 / 0.044 |
| Latency p50 | 0.1 ms | 39 ms | 1,038 ms | 20 ms |
| Cost / 1k texts | $0 | $0 | $0.169 | $0 |

(D0 @ tpr95 and D2 @ tpr95 are degenerate: FPR 1.00 and 0.77.)
D2: best on jailbreak_template (0.88 caught); 0 FPR on benign + multilingual; hard_negative
FPR 0.27, almost all from in_the_wild benign persona prompts (label conflict with SPML,
where persona redefinition is an attack). Misses: Mosscap (58 % caught) and SPML (63 %),
context-dependent attacks.

Paired: D1 − D3 AUROC +0.007 [0.005, 0.009], FPR@95 −0.028; D1 − D0 AUROC +0.327.
D1 weak spot: jailbreak_template catch 0.66 (D3: 0.77) → ensemble candidate in M5.
D1 hard_negative FPR 0.028 (D3 0.074, D0 0.217).
D3 weak spots: jailbreak_template catch 0.77; hard_negative FPR 0.074.
D0 at any_rule_fires: hard_negative FPR 0.217. Known caveats: Mosscap labels noisy;
jailbreak_template small (~1.1k attacks); deepset language = `und`.

## 7. Commands

```powershell
uv run pytest -q                          # offline tests
uv run pytest -q -m gpu                   # GPU tests
uv run ruff check . ; uv run mypy src tests
uv run gl check-env
uv run gl build-data                      # writes data/processed/<version>/
uv run gl train --detector d1             # ~23 min, data/models/d1/<run_name>/
uv run gl eval --detector all --split dev # d0, d1, d3 or all
uv run gl report --split dev              # results/*.csv + results/figures/*.png
```

## 8. Next step: M5 (full benchmark on test)

Spec: all detectors on the frozen test split **from a tagged commit** (`eval-v*`), seen vs.
held-out breakdown, ensemble, paired significance tests, FPR@TPR table, error analysis on
~50 FN + 50 FP (described by category, never quoting prompts). Acceptance =
`results/summary.csv` + headline FPR-vs-catch-rate figure.

Before anything: **git must exist** (the user said they'd add it; ask). The runner refuses
`--split test` without an `eval-v*` tag at HEAD. Make sure `.gitignore` keeps `data/`,
`.env`, `results/runs/` predictions out of the repo; first commit, then tag.
Decisions to put to the user: ensemble design (e.g. D1+D3 average, and/or D2 only on D1's
uncertain middle band; all weights/bands set on dev), which operating points to report
(tpr95 + fixed ones), D2 test cost estimate (8,212 texts ≈ $1.4; budget left ≈ $4.0),
error-analysis sampling (seeded), how to present seen vs. held-out (`adversarial_jailbreak`).
