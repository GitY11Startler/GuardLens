"""FastAPI guardrail service.

    POST /check  {"text": "...", "direction": "input"}  -> CheckResponse
    GET  /health                                        -> HealthResponse

Errors: 422 for an invalid request (empty text, direction other than "input"),
413 for a text longer than `max_chars`. `create_app` takes a ready Guard, so tests can pass
one built from fake detectors; `build_guard` loads the real models for `gl serve`.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from guardlens.service.config import ServiceConfig
from guardlens.service.guard import Guard
from guardlens.service.schemas import CheckRequest, CheckResponse, HealthResponse


def create_app(guard: Guard) -> FastAPI:
    app = FastAPI(
        title="GuardLens",
        description="Flags prompt-injection and jailbreak attempts in user input.",
        version="0.1.0",
    )
    max_chars = guard.config.max_chars

    @app.post("/check", response_model=CheckResponse)
    def check(request: CheckRequest) -> CheckResponse:
        # A plain `def`: FastAPI runs it in a worker thread, so the model call doesn't block
        # the server's event loop.
        if len(request.text) > max_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text has {len(request.text)} characters; the limit is {max_chars}",
            )
        return guard.check(request.text)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            configured_mode=guard.config.mode,
            judge_available=guard.judge is not None,
            judge_unavailable_reason=guard.judge_unavailable_reason,
        )

    return app


def build_guard(config: ServiceConfig, configs_dir: Path, data_dir: Path) -> Guard:
    """Load D1 (GPU) and, in e2 mode, the judge. A missing API key is not fatal: the guard
    then runs on D1 alone and reports why."""
    from guardlens.detectors.registry import build_detector, build_judge
    from guardlens.eval.config import load_detectors_config, load_eval_config

    detectors = load_detectors_config(configs_dir / "detectors.yaml")
    band = load_eval_config(configs_dir / "eval.yaml").ensembles["e2"].band
    d1 = build_detector("d1", detectors, [], data_dir)
    judge = None
    reason = None
    if config.mode == "e2":
        try:
            judge = build_judge(detectors, data_dir)
        except RuntimeError as e:  # e.g. ANTHROPIC_API_KEY is empty
            reason = str(e)
    return Guard(d1, judge, config, band, judge_unavailable_reason=reason)
