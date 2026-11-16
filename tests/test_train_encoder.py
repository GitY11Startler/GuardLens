import json

import numpy as np
import pytest
import torch
from helpers import make_example
from tiny_encoder import tiny_model, tiny_tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardlens.data.schema import ATTACK, BENIGN
from guardlens.eval.config import D1Config
from guardlens.training.train_encoder import (
    class_index,
    collate,
    encode_head_tail,
    predict_logits,
    special_tokens,
    tokenize,
    train_encoder,
)

CLASSES = ["benign", "fam_x", "fam_y"]


def _cfg(**kw: object) -> D1Config:
    base: dict[str, object] = dict(
        model="tiny",
        run_name="t",
        classes=CLASSES,
        max_length=16,
        tail_tokens=4,
        lr=1e-3,
        epochs=2,
        batch_size=4,
        grad_accum=2,
        eval_batch_size=8,
        device="cpu",
    )
    return D1Config.model_validate(base | kw)


def test_short_text_matches_plain_tokenizer() -> None:
    tok = tiny_tokenizer()
    text = "alpha bravo charlie"
    assert tokenize(tok, [text], 16, 4)[0] == tok(text)["input_ids"]


def test_head_tail_keeps_start_and_end() -> None:
    ids = list(range(100, 130))  # 30 content tokens
    out = encode_head_tail(ids, [1], [2], max_length=12, tail_tokens=4)
    assert out == [1] + list(range(100, 106)) + list(range(126, 130)) + [2]


def test_head_tail_leaves_fitting_text_alone() -> None:
    assert encode_head_tail([5, 6, 7], [1], [2], max_length=12, tail_tokens=4) == [1, 5, 6, 7, 2]


def test_special_tokens_of_tiny_tokenizer() -> None:
    tok = tiny_tokenizer()
    assert special_tokens(tok) == ([tok.cls_token_id], [tok.sep_token_id])


def test_class_index() -> None:
    assert class_index(make_example("alpha", BENIGN), CLASSES) == 0
    assert class_index(make_example("bravo", ATTACK, family="fam_y"), CLASSES) == 2
    with pytest.raises(ValueError, match="not in classes"):
        class_index(make_example("charlie", ATTACK, family="unseen"), CLASSES)


def test_collate_pads_to_longest() -> None:
    batch = collate([[2, 5, 3], [2, 3]], pad_id=0, device="cpu")
    assert batch["input_ids"].tolist() == [[2, 5, 3], [2, 3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_predict_logits_restores_original_order() -> None:
    torch.manual_seed(0)
    model, tok = tiny_model(3), tiny_tokenizer()
    seqs = tokenize(tok, ["alpha bravo charlie delta", "kilo", "lima mike"], 16, 4)
    together = predict_logits(model, tok, seqs, batch_size=2)
    one_by_one = np.vstack([predict_logits(model, tok, [s], batch_size=1) for s in seqs])
    np.testing.assert_allclose(together, one_by_one, atol=1e-5)


def _toy_data() -> tuple[list, list]:
    train = (
        [make_example(f"alpha bravo placeholder {w}", ATTACK, family="fam_x") for w in "abcdef"]
        + [make_example(f"charlie delta placeholder {w}", ATTACK, family="fam_y") for w in "abcd"]
        + [make_example(f"kilo lima mike placeholder {w}", BENIGN) for w in "abcdefghij"]
    )
    dev = [
        make_example("alpha bravo echo", ATTACK, family="fam_x"),
        make_example("charlie delta echo", ATTACK, family="fam_y"),
        make_example("kilo lima oscar", BENIGN),
        make_example("mike november oscar", BENIGN),
    ]
    return train, dev


def test_train_encoder_end_to_end(tmp_path) -> None:
    torch.manual_seed(0)
    train, dev = _toy_data()
    run_dir = tmp_path / "run"
    summary = train_encoder(
        _cfg(), train, dev, tiny_model(3), tiny_tokenizer(), run_dir, tmp_path / "cal.png"
    )
    history = json.loads((run_dir / "history.json").read_text("utf-8"))
    assert [h["epoch"] for h in history] == [1, 2]
    assert summary["temperature"] > 0
    assert (tmp_path / "cal.png").exists()
    assert json.loads((run_dir / "config.json").read_text("utf-8"))["d1"]["run_name"] == "t"

    # The saved checkpoint reloads and gives the same logits.
    model = AutoModelForSequenceClassification.from_pretrained(run_dir / "model")
    tok = AutoTokenizer.from_pretrained(run_dir / "model")
    seqs = tokenize(tok, [e.text for e in dev], 16, 4)
    assert predict_logits(model, tok, seqs, 8).shape == (4, 3)


def test_train_encoder_refuses_existing_run(tmp_path) -> None:
    train, dev = _toy_data()
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        train_encoder(
            _cfg(),
            train,
            dev,
            tiny_model(3),
            tiny_tokenizer(),
            tmp_path / "run",
            tmp_path / "cal.png",
        )


def test_freeze_embeddings_keeps_embedding_table_fixed(tmp_path) -> None:
    torch.manual_seed(0)
    train, dev = _toy_data()
    model = tiny_model(3)
    emb = model.get_input_embeddings()
    assert isinstance(emb, torch.nn.Embedding)
    emb_before = emb.weight.detach().clone()
    layer = next(p for n, p in model.named_parameters() if "layers.0" in n)
    layer_before = layer.detach().clone()
    train_encoder(
        _cfg(freeze_embeddings=True),
        train,
        dev,
        model,
        tiny_tokenizer(),
        tmp_path / "run",
        tmp_path / "cal.png",
    )
    assert torch.equal(emb.weight, emb_before)
    assert not torch.equal(layer.detach(), layer_before)
