"""Unified CLI for Pair Representation Scaling (β-grid sampling).

Usage:

    prs predict --model boltz2 --input seq.yaml --output out/ --beta "-0.6,0,0.6"
    prs predict --model af3    --input seq.json --output out/ --beta "-0.45,0,0.45"

Internally:
  - boltz2: runs `boltz predict` once per β value in its own process, with the
            pair representation scaling hook installed beforehand.
  - af3:    invokes `python run_alphafold.py --beta=<val>` once per β value.
            The AlphaFold 3 checkout must be patched first: `prs patch-af3`.

Each β value gets its own sub-directory under --output, named `beta_<val>`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import click

from prs import af3 as af3_patch


def _format_beta_label(value: float) -> str:
    """Return a directory-safe label, e.g. -0.45 -> beta_neg0p45, 0.30 -> beta_pos0p30."""
    abs_str = f"{abs(value):.2f}".replace(".", "p")
    sign = "neg" if value < 0 else ("pos" if value > 0 else "zero")
    return f"beta_{sign}{abs_str}" if value != 0 else "beta_zero"


def _parse_beta_list(spec: str) -> list[float]:
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise click.UsageError("--beta must contain at least one value")
    return out


def _run_boltz2(
    input_path: Path,
    output_root: Path,
    betas: Sequence[float],
    seed: int,
    samples: int,
    use_msa_server: bool,
    extra: Sequence[str],
) -> None:
    """Loop over β values, invoke `boltz predict` for each."""
    if importlib.util.find_spec("boltz") is None:
        raise click.UsageError(
            "boltz is not installed. Install it with: pip install boltz==2.2.1"
        )
    for beta in betas:
        sub = output_root / _format_beta_label(beta)
        sub.mkdir(parents=True, exist_ok=True)
        bootstrap = (
            "import prs.boltz_hook as hook; "
            f"hook.install({beta!r}); "
            "from boltz.main import cli; cli(prog_name='boltz')"
        )
        cmd = [
            sys.executable, "-c", bootstrap,
            "predict", str(input_path),
            "--out_dir", str(sub),
            "--seed", str(seed),
            "--diffusion_samples", str(samples),
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        cmd.extend(extra)
        click.echo(f"[boltz2] β={beta:+.4f} → {sub}")
        subprocess.run(cmd, check=True)


def _run_af3(
    input_path: Path,
    output_root: Path,
    betas: Sequence[float],
    samples: int,
    model_dir: str | None,
    af3_run: str | None,
    extra: Sequence[str],
) -> None:
    """Loop over β values, invoke `run_alphafold.py` for each."""
    if af3_run is None:
        repo = os.environ.get("AF3_REPO")
        if repo is None:
            raise click.UsageError(
                "Point prs at your AlphaFold 3 checkout with --af3_run or the "
                "AF3_REPO environment variable. Patch it first: prs patch-af3 <path>."
            )
        af3_run = str(Path(repo) / "run_alphafold.py")
    if not Path(af3_run).exists():
        raise click.UsageError(f"AlphaFold 3 entrypoint not found: {af3_run}")
    for beta in betas:
        sub = output_root / _format_beta_label(beta)
        sub.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, af3_run,
            f"--json_path={input_path}",
            f"--output_dir={sub}",
            f"--num_diffusion_samples={samples}",
            f"--beta={beta:.6f}",
        ]
        if model_dir is not None:
            cmd.append(f"--model_dir={model_dir}")
        cmd.extend(extra)
        click.echo(f"[af3] β={beta:+.4f} → {sub}")
        subprocess.run(cmd, check=True)


@click.group()
@click.version_option(package_name="pair-representation-scaling")
def cli() -> None:
    """Pair Representation Scaling (PRS): β-grid conformational sampling."""


@cli.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--model",
    type=click.Choice(["boltz2", "af3"]),
    required=True,
    help="Backend structure prediction model.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input file: YAML/FASTA for boltz2, JSON for af3.",
)
@click.option(
    "--output",
    "output_root",
    type=click.Path(path_type=Path),
    required=True,
    help="Output root. β sub-directories are created underneath.",
)
@click.option(
    "--beta",
    "beta_spec",
    type=str,
    default="0",
    show_default=True,
    help="Comma-separated β values, e.g. '-0.6,-0.3,0,0.3,0.6'.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed passed to the backend (boltz2 only).",
)
@click.option(
    "--samples",
    type=int,
    default=5,
    show_default=True,
    help="Diffusion samples per β value.",
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Boltz-2 only: query the ColabFold MSA server when MSA is absent.",
)
@click.option(
    "--model_dir",
    type=str,
    default=None,
    help="AF3 only: path to AlphaFold 3 model parameters directory.",
)
@click.option(
    "--af3_run",
    type=str,
    default=None,
    help=(
        "AF3 only: path to run_alphafold.py in your patched AlphaFold 3 "
        "checkout. Defaults to $AF3_REPO/run_alphafold.py."
    ),
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def predict(
    model: str,
    input_path: Path,
    output_root: Path,
    beta_spec: str,
    seed: int,
    samples: int,
    use_msa_server: bool,
    model_dir: str | None,
    af3_run: str | None,
    extra: tuple[str, ...],
) -> None:
    """Run β-grid conformational sampling. One sub-directory per β value."""
    betas = _parse_beta_list(beta_spec)
    output_root.mkdir(parents=True, exist_ok=True)

    if model == "boltz2":
        _run_boltz2(input_path, output_root, betas, seed, samples,
                    use_msa_server, list(extra))
    elif model == "af3":
        _run_af3(input_path, output_root, betas, samples,
                 model_dir, af3_run, list(extra))
    else:  # pragma: no cover
        raise click.UsageError(f"unknown model: {model}")


@cli.command("patch-af3")
@click.argument(
    "repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--patch",
    "patch_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Patch to apply. Defaults to patches/alphafold3-97639ff.patch.",
)
def patch_af3(repo: Path, patch_file: Path | None) -> None:
    """Add pair representation scaling to an AlphaFold 3 checkout."""
    try:
        patch = patch_file if patch_file is not None else af3_patch.bundled_patch()
        click.echo(af3_patch.apply_patch(repo.resolve(), patch))
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
