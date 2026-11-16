"""A tiny, randomly initialised encoder + tokenizer for offline D1 tests (no network, CPU)."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    ModernBertConfig,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

WORDS = "alpha bravo charlie delta echo kilo lima mike november oscar placeholder".split()
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def tiny_model(num_labels: int = 4) -> ModernBertForSequenceClassification:
    config = ModernBertConfig(
        vocab_size=len(SPECIALS) + len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
    )
    config.num_labels = num_labels
    return ModernBertForSequenceClassification(config)
