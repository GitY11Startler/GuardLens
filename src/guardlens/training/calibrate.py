"""Probability calibration for D1: temperature scaling on the attack logit.

The encoder outputs 4 logits (benign + 3 attack families). We fold them into one
"attack logit" m = logsumexp(attack logits) - benign logit, so sigmoid(m) = 1 - p(benign).
Calibration divides m by one temperature T > 0 fitted on dev. Dividing by a positive
constant keeps the order of the scores, so AUROC and the catch rate at a threshold rank
don't change; only how well the probabilities match reality does.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Array = np.ndarray
BENIGN_CLASS = 0  # index of the benign class in the encoder's logits
EPS = 1e-7


def attack_logit(logits: Array) -> Array:
    """(n, classes) logits -> (n,) logit of "any attack class" vs. benign."""
    attack = np.delete(logits, BENIGN_CLASS, axis=1)
    top = attack.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(attack - top).sum(axis=1))
    return lse - logits[:, BENIGN_CLASS]


def sigmoid(x: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable for large |x|


def calibrated_score(margin: Array, temperature: float) -> Array:
    return sigmoid(margin / temperature)


def log_loss(y: Array, p: Array) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(margin: Array, y: Array) -> float:
    """Temperature that minimises the log loss of sigmoid(margin / T).

    A log-spaced grid over [0.05, 20] and then a finer grid around the best point:
    plenty of precision for one parameter, and no optimiser to debug.
    """
    grid = np.geomspace(0.05, 20.0, 400)
    best = grid[int(np.argmin([log_loss(y, calibrated_score(margin, t)) for t in grid]))]
    fine = np.geomspace(best / 1.02, best * 1.02, 201)
    return float(fine[int(np.argmin([log_loss(y, calibrated_score(margin, t)) for t in fine]))])


def brier(y: Array, p: Array) -> float:
    return float(np.mean((p - y) ** 2))


def reliability_bins(y: Array, p: Array, bins: int = 15) -> list[tuple[float, float, int]]:
    """Equal-width bins over [0, 1]: (mean predicted, observed attack share, count) per
    non-empty bin."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def ece(y: Array, p: Array, bins: int = 15) -> float:
    """Expected calibration error: count-weighted gap between predicted and observed."""
    return sum(n * abs(conf - acc) for conf, acc, n in reliability_bins(y, p, bins)) / len(y)


def plot_reliability(
    y: Array, before: Array, after: Array, path: Path, title: str, bins: int = 15
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", label="perfect")
    for name, p in (("before", before), ("after", after)):
        pts = reliability_bins(y, p, bins)
        label = f"{name} (ECE {ece(y, p, bins):.3f}, Brier {brier(y, p):.3f})"
        ax.plot([c for c, _, _ in pts], [a for _, a, _ in pts], marker="o", label=label)
    ax.set(
        xlabel="Predicted attack probability",
        ylabel="Observed share of attacks",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
