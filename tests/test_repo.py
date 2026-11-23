"""Guards the git setup: no source, test or config file may be git-ignored.

(An unanchored `data/` pattern once ignored the whole src/guardlens/data/ package; it only
showed up in a fresh clone.)
"""

import shutil
import subprocess

import pytest

from guardlens.settings import PROJECT_ROOT


def test_no_project_file_is_git_ignored() -> None:
    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = [
        str(p.relative_to(PROJECT_ROOT))
        for folder in ("src", "tests", "configs", "app")
        for p in (PROJECT_ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(files),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"git-ignored project files:\n{result.stdout}"
