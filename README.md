# Pair Representation Scaling

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4-FF6F00?logo=googlecloud&logoColor=white)](https://github.com/jax-ml/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)

Pair representation scaling for
[Boltz-2](https://github.com/jwohlwend/boltz) and
[AlphaFold 3](https://github.com/google-deepmind/alphafold3). A scalar β
multiplies the pair representation, `z := (1 + β)·z`, just before the
Pairformer stack. Sweeping β at inference time produces ensembles that
recover alternative conformations of fold-switching proteins, GPCRs,
membrane transporters, and other dual-state systems.

The change is a few lines in each model, so you can write it into your own copy
instead of installing this. [`patches/`](patches/README.md) shows where those
lines go, and the CLI here runs the β sweep.

See the paper for the full method and benchmark:
[Biasing Conformational Sampling in AlphaFold 3 and Boltz-2 via Pair Representation Scaling](https://doi.org/10.1021/acs.jcim.6c02094)
(Journal of Chemical Information and Modeling, 2026).

![gpcr_si](md/muor.png)

## Installation

```bash
git clone https://github.com/suzuki-2001/pair-representation-scaling.git
cd pair-representation-scaling

# The prs CLI and Boltz-2 from PyPI, with the CUDA triangle kernels
pip install -e ".[boltz]"
```

Boltz-2 needs no patching. The CLI applies the scaling at runtime on top of a
stock `boltz` install. The `boltz` extra also installs the `cuequivariance`
triangle kernels that Boltz-2 uses on GPU. To install without them, use
`.[boltz-nokernels]` and pass `--no_kernels`.

AlphaFold 3 is built from its own source and its model parameters are requested
from Google. Patch your checkout once, then point `prs` at it:

```bash
prs patch-af3 /path/to/alphafold3
export AF3_REPO=/path/to/alphafold3
```

A Docker image is available for each backend. See [docker/](docker/README.md).

## Usage

A single `prs predict` command sweeps a β-grid for either backend.

```bash
# Boltz-2
prs predict --model boltz2 \
    --input example/rfah/boltz2_input.yaml \
    --output example/rfah/output_boltz2 \
    --beta "-0.6,-0.3,0,0.3,0.6"

# AlphaFold 3 — needs $AF3_REPO (or --af3_run) and the AF3 parameters
prs predict --model af3 \
    --input example/rfah/af3_input.json \
    --output example/rfah/output_af3 \
    --beta "-0.45,0,0.45" \
    --model_dir /path/to/af3-weights

# Single β value
prs predict --model boltz2 --input ... --output ... --beta 0.45
```

Each β value gets its own sub-directory (e.g., `output_boltz2/beta_neg0p30/`).
Setting β = 0 reproduces stock Boltz-2 or AlphaFold 3 inference.

The shipped examples declare `msa: empty` and run in single-sequence mode. Add
`--use_msa_server` to fetch an MSA from the ColabFold server for inputs that
carry none.

## Examples

Two fold-switching / activation-state benchmark systems are shipped under
[`example/`](example/README.md):

```
example/
├── rfah/   # Fold-switching protein (α-helix ↔ β-barrel)
└── muor/   # μ-opioid receptor (inactive ↔ active)
```

```bash
cd example/rfah
bash run_boltz2.sh         # Boltz-2 sweep
bash run_af3.sh            # AlphaFold 3 sweep (needs AF3_REPO and AF3_MODEL_DIR)
```

### Visualization

Each example ships a [marimo](https://marimo.io/) notebook that computes
TM-scores to each reference and plots the β-coloured scatter:

```bash
pip install marimo tmtools biopython matplotlib pandas
marimo edit example/rfah/visualize_tmscore.py
```

## Acknowledgements

This repository builds on the following projects and datasets:

- **Boltz** — Passaro *et al.* 2025, *bioRxiv*. [doi.org/10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707), and Wohlwend *et al.* 2024, *bioRxiv*. [doi.org/10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167). Primary backend, used at version 2.2.1 under the MIT License.
- **AlphaFold 3** — Abramson *et al.* 2024, *Nature*. [doi.org/10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w). Second backend, used at commit `97639ff` under CC BY-NC-SA 4.0.
- **AFsample2** — Kalakoti & Wallner 2025, *Communications Biology*. [doi.org/10.1038/s42003-025-07791-9](https://doi.org/10.1038/s42003-025-07791-9). The OC23 benchmark targets are used in our evaluations.
- **IOMemP** — Xie & Huang 2024, *Journal of Chemical Information and Modeling*. [doi.org/10.1021/acs.jcim.3c01936](https://doi.org/10.1021/acs.jcim.3c01936). Membrane-transporter dual-state references are sourced from this benchmark.

## Changelog

- [2026/8/20] Stopped vendoring Boltz-2 and AlphaFold 3. Boltz-2 comes from
  PyPI and is scaled at runtime, AlphaFold 3 is patched in place with
  `prs patch-af3`, and both changes are kept as diffs under `patches/`.
- [2026/8/20] Published in the Journal of Chemical Information and Modeling,
  [doi.org/10.1021/acs.jcim.6c02094](https://doi.org/10.1021/acs.jcim.6c02094).
- [2026/6/24] Posted the revised preprint (v2) to bioRxiv, now titled
  "Biasing Conformational Sampling in AlphaFold 3 and Boltz-2 via Pair
  Representation Scaling" and reporting AlphaFold 3 results alongside
  Boltz-2.
- [2026/5/14] Renamed the repository from `boltz-sample` to
  `pair-representation-scaling`, added AlphaFold 3 as a second backend,
  simplified the implementation to β-uniform scaling only, switched the
  user-facing CLI to `prs predict`, and shipped Docker images for both
  backends.
- [2026/1/23] Initial release as `boltz-sample`: Boltz-2-only β-uniform
  scaling with the `boltz sample` subcommand and the rfah / muor examples.

## Citation

If you use this code, please cite our paper.

```bibtex
@article{Suzuki2026PairRepresentationScaling,
    author    = {Suzuki, Shosuke and Amagasa, Toshiyuki},
    title     = {Biasing Conformational Sampling in AlphaFold 3 and Boltz-2 via Pair Representation Scaling},
    year      = {2026},
    doi       = {10.1021/acs.jcim.6c02094},
    publisher = {American Chemical Society},
    journal   = {Journal of Chemical Information and Modeling}
}
```
