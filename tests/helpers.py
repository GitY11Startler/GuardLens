from guardlens.data.schema import ATTACK, BENIGN, Example, Label, make_id


def make_example(
    text: str,
    label: Label = BENIGN,
    source: str = "toy",
    family: str = "toy_family",
) -> Example:
    """Tiny synthetic example. Texts in tests are harmless placeholders."""
    is_attack = label == ATTACK
    return Example(
        id=make_id(source, text),
        text=text,
        label=label,
        category=family if is_attack else "benign",
        family=family if is_attack else None,
        source=source,
        license="test-only",
    )
