import numpy as np
import pytest

from guardlens.env_check import check_env


@pytest.mark.gpu
def test_cuda_gpu_works() -> None:
    assert check_env() == 0


@pytest.mark.gpu
def test_real_embedding_model_loads_and_embeds() -> None:
    """Downloads the D3 model on first run (network), then embeds two placeholder texts.
    Model files go to data/raw/hf (HF_HOME is set in conftest.py)."""
    from guardlens.embeddings import SentenceTransformerEmbedder
    from guardlens.eval.config import load_detectors_config
    from guardlens.settings import get_settings

    cfg = load_detectors_config(get_settings().configs_dir / "detectors.yaml").d3
    emb = SentenceTransformerEmbedder(cfg.model, cfg.prefix, device="cuda")
    vectors = emb.embed(["a short placeholder sentence", "une phrase d'exemple"])
    assert vectors.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-4)


@pytest.mark.gpu
def test_real_encoder_forward_pass_bf16() -> None:
    """The D1 base model (untrained head) runs one bf16 forward pass on CUDA, and head+tail
    encoding of a short text matches the tokenizer's own output."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from guardlens.eval.config import load_detectors_config
    from guardlens.settings import get_settings
    from guardlens.training.train_encoder import predict_logits, tokenize

    cfg = load_detectors_config(get_settings().configs_dir / "detectors.yaml").d1
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, num_labels=len(cfg.classes)
    ).to("cuda")
    texts = ["a short placeholder sentence", "une phrase d'exemple " * 400]
    seqs = tokenize(tok, texts, cfg.max_length, cfg.tail_tokens)
    assert seqs[0] == tok(texts[0])["input_ids"]
    assert len(seqs[1]) == cfg.max_length
    logits = predict_logits(model, tok, seqs, batch_size=2)
    assert logits.shape == (2, len(cfg.classes)) and np.all(np.isfinite(logits))
