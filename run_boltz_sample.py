#!/usr/bin/env python3
"""Deprecated: Use 'boltz sample' command instead."""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import click
import yaml

from boltz.console import info, success, error, warn

warnings.warn(
    "run_boltz_sample.py is deprecated. Use 'boltz sample' command instead.",
    DeprecationWarning,
    stacklevel=1,
)


def extract_sequence(fasta: Path) -> str:
    """Extract sequence from FASTA."""
    return "".join(
        line.strip()
        for line in fasta.open()
        if line.strip() and not line.startswith(("#", ">"))
    )


def make_yaml(msa: Path, seq: str, out_yaml: Path) -> None:
    """Create YAML recipe."""
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seq, "msa": str(msa.resolve())}}
        ],
    }
    with out_yaml.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


def run_boltz(
    yaml_path: Path,
    out_dir: Path,
    num_workers: int,
    diffusion_samples: int,
    seed: int,
    no_msa: bool,
    max_msa_seqs: int | None,
    extra: Iterable[str],
) -> None:
    """Execute boltz command."""
    num_msa = 0 if no_msa else (50000 if max_msa_seqs is None else max_msa_seqs)
    cmd = [
        "boltz", "predict", str(yaml_path),
        "--model", "boltz2",
        "--num_workers", str(num_workers),
        "--diffusion_samples", str(diffusion_samples),
        "--out_dir", str(out_dir),
        "--max_msa_seqs", str(num_msa),
        "--devices", "1",
        "--seed", str(seed),
    ]
    cmd.extend(extra)
    subprocess.run(cmd, check=True)


def rate_str(x: float) -> str:
    """Convert float to compact string."""
    s = f"{x:.2f}".replace(".", "p")
    return s.rstrip("0").rstrip("p")


def build_uniform(betas: Iterable[float]) -> List[Tuple[str, List[str]]]:
    """Build uniform scaling configs."""
    return [
        (f"scaleuniform_{rate_str(b)}", ["--scale_uniform_beta", str(b)])
        for b in betas if abs(b) > 1e-8
    ]


@click.command()
@click.option("--fasta", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--msa", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--yaml_dir", type=click.Path(path_type=Path), required=True)
@click.option("--out_dir", type=click.Path(path_type=Path), required=True)
@click.option("--num_workers", default=8, show_default=True)
@click.option("--diffusion_samples", default=10, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--no_msa", is_flag=True, help="Skip MSA step.")
@click.option("--max_msa_seqs", type=int, default=None, help="Set max MSA sequences.")
@click.option(
    "--scale_uniform_beta",
    default="-0.15,-0.30,-0.45,-0.60,-0.75,0.15,0.30,0.45,0.60,0.75",
    show_default=True,
    help="Comma-separated β values for uniform scaling.",
)
@click.argument("extra_args", nargs=-1)
def main(
    fasta: Path,
    msa: Path,
    yaml_dir: Path,
    out_dir: Path,
    num_workers: int,
    diffusion_samples: int,
    seed: int,
    no_msa: bool,
    max_msa_seqs: int | None,
    scale_uniform_beta: str,
    extra_args: Tuple[str, ...],
) -> None:
    """Boltz2 prediction runner for uniform scaling."""
    warn("This script is deprecated. Use 'boltz sample' command instead.")
    info("Boltz2 uniform scaling runner started")

    yaml_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = extract_sequence(fasta)
    betas = [float(x) for x in scale_uniform_beta.split(",") if x.strip()]

    for suffix, extra in build_uniform(betas):
        yml = yaml_dir / f"{msa.stem}_{suffix}.yaml"
        run_path = out_dir / suffix

        if not yml.exists():
            info(f"Generate YAML: {yml.name}")
            make_yaml(msa, seq, yml)

        run_path.mkdir(parents=True, exist_ok=True)
        info(f"Run Boltz ({suffix})")

        try:
            run_boltz(
                yml,
                run_path,
                num_workers,
                diffusion_samples,
                seed,
                no_msa,
                max_msa_seqs,
                extra_args + tuple(extra),
            )
        except subprocess.CalledProcessError as err:
            error(f"Boltz failed for {suffix}: {err}")
            continue

        success(f"Finished {suffix}")


if __name__ == "__main__":
    main()
