# GuardLens — Project Specification

> A defensive guardrail for LLM applications that **detects** prompt-injection and jailbreak attempts (and unsafe outputs), plus a rigorous **benchmark** that measures how well different guardrail strategies work.
> Owner: Yassine Zanned · Status: spec v1 (2026-11-06) · Target: public GitHub portfolio project

---

## 0. Scope & ethics (read first)

**What this project is:** a *detector / filter* — the blue-team component that sits in front of (and behind) an LLM and flags malicious inputs and unsafe outputs. It is evaluated on **existing, publicly released** attack datasets used as test material.

**What this project is NOT, by design:**
- It does **not** generate, mutate, or optimize new attacks/jailbreaks.
- It does **not** include an "attacker agent."
- It contains no catalogue of working jailbreak techniques.

If a future idea drifts toward "automatically find prompts that break model X," it's out of scope — that's the line this project deliberately stays on the safe side of. Keeping the project purely defensive is also what makes it clean to put on a public GitHub and talk about in interviews.

---

## 1. Problem & pitch

Any app that puts user text in front of an LLM (chatbots, RAG, tool-using agents) can be steered by **prompt injection** ("ignore your instructions and …") or **jailbreaks** (framings that elicit disallowed output). Teams defend with a **guardrail**: a fast classifier (and/or an LLM judge) that screens inputs before they reach the model and screens outputs before they reach the user.

**GuardLens** builds that guardrail and — more importantly for a portfolio — a **fair benchmark** answering: how well do different guardrail strategies catch attacks, at what false-positive cost, at what latency and price?

**Headline deliverables:**
1. A deployable guardrail service (allow / block + reason + score).
2. A reproducible benchmark comparing guardrail strategies with proper statistics.
3. An honest analysis: where detectors fail, and the accuracy/latency/cost trade-off.

**What it demonstrates:** LLM orchestration, classifier fine-tuning, deterministic evaluation (your signature), and an understanding of AI security from the defender's side — all of which are directly hireable.

---

## 2. System overview

```
                    ┌──────────────── DATA LAYER ────────────────┐
 public datasets →  │ loaders → unified schema {text, label,     │
 (benign + attack)  │ category, source} → dedup → train/dev/test │
                    └───────────────────────► data/processed/ ───┘

                    ┌──────────────── DETECTORS (defenses) ──────┐
 input text    →    │ D0 heuristic baseline (rules/keywords)     │ → score ∈ [0,1]
                    │ D1 fine-tuned encoder (DeBERTa/ModernBERT) │   + label
                    │ D2 LLM-as-judge (local or API, rubric)     │   + reason
                    │ D3 embedding k-NN to known-attack corpus   │
                    └───────────────────────┬────────────────────┘
                                            │ calibrated threshold
                    ┌───────────────────────▼─────────────────────┐
 EVAL   →           │ precision/recall/F1, AUROC, FPR@TPR,        │
                    │ per-category, latency, cost, CIs            │
                    └─────────────────────────────────────────────┘

 SERVE  →  FastAPI /check {text, direction=input|output} → {allow, score, category, reason}
 DEMO   →  Gradio: paste a prompt, see every detector's verdict side by side
```

**Two directions, same framework:**
- **Input screening:** is this user text a prompt-injection / jailbreak attempt?
- **Output screening (optional, phase 2):** is this model output unsafe? (uses public safety-moderation datasets)

---

## 3. Data

Use **only published, properly licensed datasets**, loaded by scripts that download by name (never vendored into git). Record each example's `source` and `license`. Candidate public sources (verify availability + license before use):

- Prompt-injection labeled sets (e.g. `deepset/prompt-injections` and similar on Hugging Face).
- Jailbreak-prompt collections released for research (e.g. "in-the-wild jailbreak prompts" datasets).
- Safety-moderation datasets for output screening (e.g. ToxicChat, OpenAI moderation set, WildGuardMix, Aegis — whichever are license-compatible).
- **Benign/hard-negative** text is essential: ordinary instructions, role-play that is *not* an attack, security questions asked legitimately, multilingual text. Without strong negatives the benchmark is meaningless (everything looks like an attack).

**Splitting rule that makes the project credible:** split by **source and attack family**, and keep at least one or two **attack categories entirely held out** of training (a "novel attack" test set). This measures generalization to attacks the detector never saw — the question defenders actually care about. Fixed seeds, splits saved as JSON, documented class balance.

---

## 4. Detectors (the defenses you build)

| ID | Strategy | Notes | GPU |
|---|---|---|---|
| D0 | Heuristic baseline | transparent rules / keyword & pattern flags | none |
| D1 | Fine-tuned encoder classifier | DeBERTa-v3-base or ModernBERT; the workhorse | fits 8 GB easily |
| D2 | LLM-as-judge | small local model (Qwen3-4B/Llama-3.1-8B-Q4) or API, scores against a written rubric | local 4-bit or API |
| D3 | Embedding k-NN | sentence embeddings + FAISS; flag inputs close to known-attack cluster | CPU/GPU light |

Each detector implements the same interface: `score(text, direction) -> {score∈[0,1], category, reason}`. That uniformity is what lets the benchmark be fair, and it mirrors the backend-abstraction pattern you used in VideoSleuth.

Optional **ensemble**: combine D1+D3 (and D2 on the uncertain middle band) to trade cost for accuracy — a nice result to show.

---

## 5. Evaluation protocol

- **Metrics:** precision, recall, F1, **AUROC**, and **FPR at a fixed TPR** (e.g. FPR @ 95% TPR) — the operating point defenders care about. Per-attack-category breakdown. **Latency** (p50/p95) and **cost per 1k requests** for each detector.
- **The decisive view:** detectors are threshold-based, so report **ROC / precision-recall curves**, not a single number. The headline figure is a **false-positive-rate vs. catch-rate** plot across all detectors.
- **Generalization test:** report metrics separately on *seen* vs. *held-out* attack families. Expect a real drop on held-out — reporting it honestly is the point.
- **Statistics:** 95% bootstrap confidence intervals; paired bootstrap / McNemar when comparing two detectors on the same test set.
- **Error analysis (manual, ~50 each way):** false negatives (attacks that slipped through, described at the category level — *not* reproduced verbatim) and false positives (benign text wrongly blocked — usually the more business-critical failure).

**Realistic expectations:**
- A fine-tuned encoder typically scores high on *in-distribution* attacks and noticeably lower on *held-out* families — that gap is your most interesting finding.
- False positives on benign-but-edgy text (security questions, fiction, role-play) are the hard part and the most valuable thing to characterize.
- LLM-as-judge is usually strong but slow and costly; quantifying that trade-off is a concrete, resume-worthy result.

---

## 6. Hardware & resources

**Target:** RTX 5070, 8 GB VRAM, 32 GB RAM, WSL2/Linux. (Same Blackwell/CUDA-≥12.8 note as VideoSleuth — verify the PyTorch build in setup.)

**VRAM:** comfortable. Encoder fine-tuning (D1) on base-size models fits in a few GB. D2 local judge uses a 4-bit 4–8B model (~3–6 GB), run as a separate stage, not alongside training. No model needs the full 8 GB.

**Disk:** small. These are text datasets.

| Item | Size |
|---|---|
| Python env + PyTorch/CUDA | 6–10 GB |
| Datasets (text) | < 2 GB |
| Models (encoder, embeddings, optional local judge 4-bit) | 5–12 GB |
| Results / logs / FAISS index | < 1 GB |
| **Total** | **≈ 15–25 GB** |

**Compute time:** encoder fine-tune is minutes to ~1–2 h per run; full benchmark across detectors is minutes (D0/D1/D3) plus whatever D2 costs. Everything is cacheable and resumable.

**Cost:** ~€0 if you stay local. If you use an API model for D2, log tokens from day one, estimate on dev first, set a hard cap — a few euros covers the whole project at these dataset sizes.

---

## 7. Tech stack

Python 3.12, env via `uv`.

| Purpose | Library |
|---|---|
| DL runtime | `torch` (CUDA ≥ 12.8), `transformers`, `accelerate`, `datasets`, `bitsandbytes` |
| Encoder training | `transformers` Trainer or `sentence-transformers`; `scikit-learn` for metrics/calibration |
| Embeddings / k-NN | `sentence-transformers` (or SigLIP-text/E5), `faiss-cpu` |
| LLM judge | `anthropic` (API) and/or OpenAI-compatible local server (llama.cpp/Ollama) via `openai` |
| Config / schemas | `pydantic`, `pydantic-settings`, YAML |
| Eval & plots | `numpy`, `scipy`, `pandas`, `matplotlib` |
| Serve | `fastapi`, `uvicorn` |
| Demo | `gradio` |
| Quality | `pytest`, `ruff`, `mypy` (lenient), `pre-commit` |

---

## 8. Repository layout

```
guardlens/
├── CLAUDE.md                 # working rules for Claude Code
├── PROJECT_SPEC.md           # this file
├── README.md                 # pitch, results curves, demo GIF, how to reproduce
├── pyproject.toml
├── configs/
│   ├── data.yaml             # dataset sources, licenses, split policy, held-out families
│   ├── detectors.yaml        # per-detector config, thresholds
│   └── eval.yaml             # systems, metrics, seeds
├── data/                     # git-ignored (raw/, processed/, splits/, cache/)
├── src/guardlens/
│   ├── data/                 # loaders.py (one per source), schema.py, build_dataset.py, splits.py
│   ├── detectors/            # base.py, d0_heuristic.py, d1_encoder.py, d2_judge.py, d3_knn.py, ensemble.py
│   ├── training/             # train_encoder.py, calibrate.py
│   ├── eval/                 # runner.py, metrics.py, stats.py, curves.py, error_analysis.py
│   ├── service/              # app.py (FastAPI /check), schemas.py
│   └── cli.py                # `gl build-data`, `gl train`, `gl eval`, `gl serve`, `gl report`
├── scripts/                  # download_datasets.py, make_splits.py, plot_curves.py
├── app/gradio_app.py         # paste text → every detector's verdict + score
├── tests/                    # unit tests on a tiny synthetic labeled set (no network/GPU)
└── results/                  # committed: summary CSVs + figures (no raw data)
```

---

## 9. Milestones (each with an acceptance check)

**M0 — Setup (½ wk):** repo, `uv` env, GPU/CUDA check script. ✅ `pytest` + ruff pass.

**M1 — Data layer (wk 1):** loaders → unified schema → dedup → source/family-aware splits with a held-out attack family; class-balance report. ✅ `gl build-data` produces versioned train/dev/test JSON with documented sources & licenses.

**M2 — Baselines D0 + D3 (wk 1–2):** heuristic + embedding k-NN, full metrics on dev. ✅ ROC/PR curves and a results CSV exist.

**M3 — Encoder D1 (wk 2–3):** fine-tune DeBERTa/ModernBERT; probability calibration; threshold selection on dev. ✅ beats D0 on dev with CIs; calibration plot saved.

**M4 — LLM judge D2 (wk 3–4):** rubric prompt, local and/or API backend, caching. ✅ runs on dev with latency + cost logged.

**M5 — Full benchmark + generalization (wk 4–5):** all detectors on frozen test (from a tagged commit), seen vs. held-out breakdown, ensemble, paired significance tests, FPR@TPR table, error analysis on ~50 FN + 50 FP. ✅ `results/summary.csv` + headline FPR-vs-catch-rate figure.

**M6 — Serve + demo + README (wk 5–6):** FastAPI `/check`, Gradio verdict viewer, README with curves, limitations, reproduce steps, demo GIF. ✅ fresh clone + `uv sync` + documented commands reproduces dev results.

**Stretch:** output-moderation direction (phase 2), multilingual slice (you have Arabic/French/English — a genuinely novel angle, since most guardrails are English-only), distillation of D2's labels into D1, Docker image.

---

## 10. Résumé bullet (template; fill in real numbers at the end)

> **GuardLens – LLM guardrail & evaluation benchmark** (PyTorch, Transformers, FAISS, FastAPI). Built and benchmarked prompt-injection / jailbreak **detectors** (fine-tuned encoder, LLM-as-judge, embedding k-NN) on public datasets with source/family-aware splits; reported catch-rate vs. false-positive-rate, latency and cost, and measured generalization to held-out attack families (F1 X% in-distribution → Y% held-out). Shipped a FastAPI guardrail service; all models run on an 8 GB GPU.
```
