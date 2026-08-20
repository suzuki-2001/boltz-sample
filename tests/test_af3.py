"""Tests for the AlphaFold 3 patch helper."""

import subprocess

import pytest

from prs import af3


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_bundled_patch_exists():
    assert af3.bundled_patch().name == af3.PATCH_NAME


def test_rejects_a_directory_without_the_entrypoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_alphafold.py"):
        af3.apply_patch(tmp_path, af3.bundled_patch())


def test_apply_then_detect_already_applied(tmp_path):
    _git(tmp_path, "init", "--quiet")
    target = tmp_path / "run_alphafold.py"
    target.write_text("value = 1\n")
    patch = tmp_path / "add-line.patch"
    patch.write_text(
        "--- a/run_alphafold.py\n"
        "+++ b/run_alphafold.py\n"
        "@@ -1 +1,2 @@\n"
        " value = 1\n"
        "+extra = 2\n"
    )

    assert "applied" in af3.apply_patch(tmp_path, patch)
    assert target.read_text() == "value = 1\nextra = 2\n"

    assert af3.is_applied(tmp_path, patch)
    assert "already applied" in af3.apply_patch(tmp_path, patch)
