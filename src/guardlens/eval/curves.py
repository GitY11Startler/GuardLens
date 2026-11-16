"""ROC, precision-recall and FPR-vs-catch-rate plots, one line per detector."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

Curves = dict[str, tuple[np.ndarray, np.ndarray]]  # detector -> (y, s)


def plot_all(curves: Curves, out_dir: Path, split: str, target_tpr: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set(xlabel="False-positive rate", ylabel="Catch rate (TPR)", title=f"ROC ({split})")
    ax.legend(loc="lower right")
    paths.append(_save(fig, out_dir / f"roc_{split}.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, s) in curves.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall ({split})")
    ax.legend(loc="lower left")
    paths.append(_save(fig, out_dir / f"pr_{split}.png"))

    # Headline view: how many benign requests get blocked for a given catch rate.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(tpr, np.clip(fpr, 1e-4, 1), label=name)
    ax.axvline(target_tpr, color="grey", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set(
        xlabel="Catch rate (share of attacks flagged)",
        ylabel="False-positive rate (log)",
        title=f"False positives vs. catch rate ({split})",
    )
    ax.legend(loc="upper left")
    paths.append(_save(fig, out_dir / f"fpr_vs_catch_{split}.png"))
    return paths


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
