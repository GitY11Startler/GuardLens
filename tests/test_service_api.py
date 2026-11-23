"""The /check API through FastAPI's test client, with fake detectors (no GPU or network)."""

from fake_detectors import Fixed
from fastapi.testclient import TestClient

from guardlens.service.app import create_app
from guardlens.service.config import ServiceConfig, Thresholds
from guardlens.service.guard import Guard

CONFIG = ServiceConfig(mode="e2", thresholds=Thresholds(d1=0.7, e2=0.4), max_chars=50)


def _client(d1_score: float, judge: Fixed | None = None, reason: str | None = None) -> TestClient:
    guard = Guard(Fixed("d1", d1_score), judge, CONFIG, (0.05, 0.95), reason)
    return TestClient(create_app(guard))


def test_check_blocks_and_allows() -> None:
    blocked = _client(0.99).post("/check", json={"text": "placeholder"})
    assert blocked.status_code == 200
    body = blocked.json()
    assert body["allow"] is False and body["decided_by"] == "d1"
    assert set(body) == {
        "allow", "score", "threshold", "category", "reason",
        "decided_by", "mode", "fallback", "latency_ms",
    }  # fmt: skip
    allowed = _client(0.01).post("/check", json={"text": "placeholder", "direction": "input"})
    assert allowed.json()["allow"] is True


def test_check_uses_the_judge_when_unsure() -> None:
    body = _client(0.5, Fixed("d2", 1.0)).post("/check", json={"text": "placeholder"}).json()
    assert body["decided_by"] == "d2" and body["allow"] is False


def test_output_direction_is_rejected_with_422() -> None:
    r = _client(0.1).post("/check", json={"text": "placeholder", "direction": "output"})
    assert r.status_code == 422
    assert "only direction='input' is supported" in r.text


def test_empty_or_missing_text_is_422() -> None:
    client = _client(0.1)
    assert client.post("/check", json={"text": ""}).status_code == 422
    assert client.post("/check", json={}).status_code == 422


def test_too_long_text_is_413() -> None:
    r = _client(0.1).post("/check", json={"text": "x" * 51})
    assert r.status_code == 413
    assert "limit is 50" in r.json()["detail"]
    assert _client(0.1).post("/check", json={"text": "x" * 50}).status_code == 200


def test_health_reports_judge_state() -> None:
    ok = _client(0.1, Fixed("d2", 0.0)).get("/health").json()
    assert ok == {
        "status": "ok",
        "configured_mode": "e2",
        "judge_available": True,
        "judge_unavailable_reason": None,
    }
    missing = _client(0.1, None, "no API key").get("/health").json()
    assert missing["judge_available"] is False
    assert missing["judge_unavailable_reason"] == "no API key"
