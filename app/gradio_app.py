"""GuardLens demo: every detector's verdict side by side, in the browser (local only).

    uv run --group demo python app/gradio_app.py      ->  http://127.0.0.1:7860

Loads D0, D1 (GPU), D3 (embedding model on the CPU, so only one heavy model is on the GPU)
and the LLM judge if ANTHROPIC_API_KEY is set. The judge only runs when its box is ticked.
"""

from guardlens.cli import use_project_hf_cache


def main() -> None:
    use_project_hf_cache()  # before any Hugging Face import

    from guardlens.data.build_dataset import load_split
    from guardlens.detectors.base import Detector
    from guardlens.detectors.registry import build_detector, build_judge
    from guardlens.eval.config import load_detectors_config, load_eval_config
    from guardlens.service.config import load_service_config
    from guardlens.service.demo import build_ui, compare
    from guardlens.settings import get_settings

    settings = get_settings()
    service = load_service_config(settings.configs_dir / "service.yaml")
    eval_cfg = load_eval_config(settings.configs_dir / "eval.yaml")
    det_cfg = load_detectors_config(settings.configs_dir / "detectors.yaml")
    det_cfg = det_cfg.model_copy(
        update={"d3": det_cfg.d3.model_copy(update={"device": service.d3_device})}
    )
    band = eval_cfg.ensembles["e2"].band

    train = load_split(settings.data_dir / "processed", eval_cfg.data_version, "train")
    detectors: dict[str, Detector] = {
        name: build_detector(name, det_cfg, train, settings.data_dir) for name in ("d0", "d1", "d3")
    }
    try:
        detectors["d2"] = build_judge(det_cfg, settings.data_dir)
    except RuntimeError as e:
        print(f"[demo] LLM judge unavailable: {e}")

    def run(text: str, use_judge: bool) -> list[list[object]]:
        if not text.strip():
            return []
        text = text[: service.max_chars]
        return compare(text, detectors, service.thresholds, band, use_judge)

    build_ui(run).launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
