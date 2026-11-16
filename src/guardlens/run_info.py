"""Provenance helpers recorded with every build / run."""

import hashlib
import subprocess
from pathlib import Path

from guardlens.settings import PROJECT_ROOT


def git_commit() -> str:
    """Current commit hash, or "no-git" while the project is not a git repository yet."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "no-git"
    return result.stdout.strip()


def eval_tag_at_head() -> str | None:
    """The `eval-v*` tag pointing at the current commit, if any."""
    try:
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags", "--match", "eval-v*", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def working_tree_clean() -> bool:
    """True when git sees no uncommitted changes outside `results/` (runs write there while
    a test evaluation is in progress). False when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return result.stdout.strip() == ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
