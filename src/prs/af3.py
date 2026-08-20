"""Apply the pair representation scaling patch to an AlphaFold 3 checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path

PATCH_NAME = "alphafold3-97639ff.patch"


def bundled_patch() -> Path:
    """Return the path to the AlphaFold 3 patch shipped with this repository."""
    here = Path(__file__).resolve()
    for root in (here.parents[2], here.parent):
        path = root / "patches" / PATCH_NAME
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{PATCH_NAME} not found next to the prs package. Run this from a "
        "checkout of pair-representation-scaling, or pass --patch."
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def is_applied(repo: Path, patch: Path) -> bool:
    """Return True when the patch is already present in `repo`."""
    return _git(repo, "apply", "-p1", "--reverse", "--check", str(patch)).returncode == 0


def apply_patch(repo: Path, patch: Path) -> str:
    """Apply the patch to an AlphaFold 3 checkout and return a status line."""
    entrypoint = repo / "run_alphafold.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"{repo} does not look like an AlphaFold 3 checkout "
            "(run_alphafold.py is missing)."
        )

    if is_applied(repo, patch):
        return f"{patch.name} is already applied to {repo}"

    result = _git(repo, "apply", "-p1", str(patch))
    if result.returncode == 0:
        return f"applied {patch.name} to {repo}"

    fallback = subprocess.run(
        ["patch", "-p1", "-i", str(patch)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if fallback.returncode == 0:
        return f"applied {patch.name} to {repo}"

    raise RuntimeError(
        f"could not apply {patch.name} to {repo}.\n"
        f"git apply: {result.stderr.strip()}\n"
        f"patch: {fallback.stderr.strip() or fallback.stdout.strip()}"
    )
