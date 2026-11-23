"""One loader per data source. Each downloads the dataset from the Hugging Face Hub
(cached under `data/raw/hf/`) and maps it to the unified `Example` schema.

All splits of a source are loaded and pooled: we re-split ourselves after dedup.
Only the fields we need are kept; which rows count as attacks is decided here, per source.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset

from guardlens.data.config import SourceConfig
from guardlens.data.schema import ATTACK, BENIGN, Example, Label, make_id

Rows = Iterable[dict[str, Any]]


class LoaderContext:
    """What every loader needs: where to cache, and a token for gated datasets."""

    def __init__(self, cache_dir: Path, hf_token: str | None) -> None:
        self.cache_dir = cache_dir
        self.hf_token = hf_token


def _rows(
    ctx: LoaderContext, cfg: SourceConfig, config_name: str | None = None, **load_kwargs: Any
) -> Rows:
    if cfg.gated and not ctx.hf_token:
        raise RuntimeError(
            f"{cfg.hf_id} is gated: accept its terms on huggingface.co and set HF_TOKEN in .env"
        )
    ds = load_dataset(
        cfg.hf_id,
        name=config_name,
        cache_dir=str(ctx.cache_dir),
        token=ctx.hf_token,
        **load_kwargs,
    )
    return concatenate_datasets(list(ds.values()))  # type: ignore[union-attr]


def _example(
    name: str,
    cfg: SourceConfig,
    text: str | None,
    label: Label,
    category: str = "benign",
    language: str = "en",
    hard_negative: bool = False,
) -> Example | None:
    if text is None or not text.strip():
        return None
    return Example(
        id=make_id(name, text),
        text=text,
        label=label,
        category=cfg.family if label == ATTACK and cfg.family else category,
        family=cfg.family if label == ATTACK else None,
        source=name,
        license=cfg.license,
        language=language,
        hard_negative=hard_negative,
    )


def _keep(examples: Iterable[Example | None]) -> list[Example]:
    return [e for e in examples if e is not None]


# --- prompt-injection sources -----------------------------------------------------------


def load_deepset(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # label: 1 = injection, 0 = benign. Mix of English and German, no language column.
    return _keep(
        _example(name, cfg, r["text"], ATTACK if r["label"] == 1 else BENIGN, language="und")
        for r in _rows(ctx, cfg)
    )


def load_gandalf(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Every row is an "ignore your instructions"-style attempt.
    return _keep(_example(name, cfg, r["text"], ATTACK) for r in _rows(ctx, cfg))


def load_mosscap(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Every row is a player's attempt to extract the secret password.
    return _keep(_example(name, cfg, r["prompt"], ATTACK) for r in _rows(ctx, cfg))


def load_spml(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Only the user turn is screened; the system prompt is the app's own context.
    return _keep(
        _example(name, cfg, r["User Prompt"], ATTACK if r["Prompt injection"] == 1 else BENIGN)
        for r in _rows(ctx, cfg)
    )


# --- jailbreak sources ------------------------------------------------------------------


def load_in_the_wild(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Jailbreak prompts and ordinary prompts from the same communities (two snapshots each;
    # overlap between snapshots is removed by dedup). Ordinary prompts are hard negatives.
    examples: list[Example | None] = []
    for config_name in (
        "jailbreak_2023_05_07",
        "jailbreak_2023_12_25",
        "regular_2023_05_07",
        "regular_2023_12_25",
    ):
        for r in _rows(ctx, cfg, config_name):
            is_attack = bool(r["jailbreak"])
            examples.append(
                _example(
                    name,
                    cfg,
                    r["prompt"],
                    ATTACK if is_attack else BENIGN,
                    category="hard_negative",
                    hard_negative=not is_attack,
                )
            )
    return _keep(examples)


def load_jackhhao(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Benign side is largely "act as ..." role-play prompts, so it counts as hard negatives.
    return _keep(
        _example(
            name,
            cfg,
            r["prompt"],
            ATTACK if r["type"] == "jailbreak" else BENIGN,
            category="hard_negative",
            hard_negative=r["type"] != "jailbreak",
        )
        for r in _rows(ctx, cfg)
    )


def load_wildjailbreak(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Keep only adversarial rows: adversarial_harmful = jailbreak (attack),
    # adversarial_benign = benign look-alike (hard negative). Vanilla rows are plain
    # requests without manipulation and are out of scope for input screening.
    # TSV files: empty cells must stay "" (not NaN), otherwise column types are mis-inferred.
    examples: list[Example | None] = []
    for config_name in ("train", "eval"):
        for r in _rows(ctx, cfg, config_name, delimiter="\t", keep_default_na=False):
            if r["data_type"] == "adversarial_harmful":
                examples.append(_example(name, cfg, r["adversarial"], ATTACK))
            elif r["data_type"] == "adversarial_benign":
                examples.append(
                    _example(
                        name, cfg, r["adversarial"], BENIGN, "hard_negative", hard_negative=True
                    )
                )
    return _keep(examples)


# --- benign / hard-negative sources -----------------------------------------------------


def load_xstest(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Only the "safe" prompts that resemble unsafe ones; the unsafe ones are plain
    # harmful requests (out of scope).
    return _keep(
        _example(name, cfg, r["prompt"], BENIGN, "hard_negative", hard_negative=True)
        for r in _rows(ctx, cfg)
        if r["label"] == "safe"
    )


def load_orbench_hard(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    return _keep(
        _example(name, cfg, r["prompt"], BENIGN, "hard_negative", hard_negative=True)
        for r in _rows(ctx, cfg, "or-bench-hard-1k")
    )


def load_orbench_80k(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    return _keep(
        _example(name, cfg, r["prompt"], BENIGN, "hard_negative", hard_negative=True)
        for r in _rows(ctx, cfg, "or-bench-80k")
    )


def load_dolly(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Instruction only; the optional reference context is not part of the user's request.
    return _keep(_example(name, cfg, r["instruction"], BENIGN) for r in _rows(ctx, cfg))


AYA_LANGUAGES = {"arb": "ar", "fra": "fr", "spa": "es", "deu": "de"}


def load_aya(name: str, cfg: SourceConfig, ctx: LoaderContext) -> list[Example]:
    # Multilingual benign prompts. German is included so that "German" is not a shortcut
    # for "attack" (deepset contains German injections).
    return _keep(
        _example(
            name, cfg, r["inputs"], BENIGN, "multilingual", AYA_LANGUAGES[r["language_code"]]
        )
        for r in _rows(ctx, cfg)
        if r["language_code"] in AYA_LANGUAGES
    )


Loader = Callable[[str, SourceConfig, LoaderContext], list[Example]]

# Keys must match the source names in configs/data.yaml.
LOADERS: dict[str, Loader] = {
    "deepset": load_deepset,
    "gandalf": load_gandalf,
    "mosscap": load_mosscap,
    "spml": load_spml,
    "in_the_wild": load_in_the_wild,
    "jackhhao": load_jackhhao,
    "wildjailbreak": load_wildjailbreak,
    "xstest": load_xstest,
    "orbench_hard": load_orbench_hard,
    "orbench_80k": load_orbench_80k,
    "dolly": load_dolly,
    "aya": load_aya,
}
