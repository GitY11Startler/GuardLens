"""Build a detector by name from the configs."""

from pathlib import Path

from guardlens.data.schema import Example
from guardlens.detectors.base import Detector
from guardlens.detectors.d0_heuristic import HeuristicDetector
from guardlens.detectors.d1_encoder import EncoderDetector
from guardlens.detectors.d2_judge import JudgeDetector
from guardlens.detectors.d3_knn import KNNDetector
from guardlens.embeddings import CachedEmbedder, SentenceTransformerEmbedder
from guardlens.eval.config import DetectorsConfig
from guardlens.judge_cost import SpendLedger
from guardlens.settings import get_settings

DETECTOR_NAMES = ["d0", "d1", "d2", "d3"]


def build_detector(
    name: str, config: DetectorsConfig, train: list[Example], data_dir: Path
) -> Detector:
    if name == "d0":
        return HeuristicDetector()
    if name == "d1":
        c1 = config.d1
        return EncoderDetector.load(
            d1_run_dir(data_dir, c1.run_name), c1.device, c1.eval_batch_size
        )
    if name == "d2":
        return build_judge(config, data_dir)
    if name == "d3":
        c = config.d3
        embedder = SentenceTransformerEmbedder(
            c.model, c.prefix, c.batch_size, c.max_seq_length, c.device
        )
        return KNNDetector(CachedEmbedder(embedder, data_dir / "cache" / "embeddings"), k=c.k).fit(
            train
        )
    raise ValueError(f"unknown detector {name!r}; choose from {DETECTOR_NAMES}")


def build_judge(config: DetectorsConfig, data_dir: Path) -> JudgeDetector:
    import anthropic

    key = get_settings().anthropic_api_key
    if key is None or not key.get_secret_value().strip():
        raise RuntimeError("ANTHROPIC_API_KEY is empty: add it to .env to use d2")
    c = config.d2
    client = anthropic.Anthropic(api_key=key.get_secret_value().strip(), max_retries=c.max_retries)
    judge_dir = data_dir / "cache" / "judge"
    return JudgeDetector(
        client,
        c.model,
        c.effort,
        c.max_tokens,
        c.prices,
        SpendLedger(judge_dir / "spend.jsonl", config.spending_cap_usd),
        judge_dir / "answers",
        c.concurrency,
    )


def d1_run_dir(data_dir: Path, run_name: str) -> Path:
    return data_dir / "models" / "d1" / run_name
