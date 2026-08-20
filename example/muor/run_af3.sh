#!/bin/bash
# μ-Opioid Receptor (GPCR) — AlphaFold 3 β-grid sampling
# References: 4DKL (inactive, antagonist-bound), 5C1M (active, agonist-bound)
# Requires: a patched AlphaFold 3 checkout at $AF3_REPO (prs patch-af3 <path>)
#           and AlphaFold 3 model weights (request from Google) at $AF3_MODEL_DIR

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AF3_REPO:?Set AF3_REPO to your patched AlphaFold 3 checkout}"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to your AlphaFold 3 model parameters directory}"

prs predict \
    --model af3 \
    --input "$SCRIPT_DIR/af3_input.json" \
    --output "$SCRIPT_DIR/output_af3" \
    --beta "-0.6,-0.3,0,0.3,0.6" \
    --samples 5 \
    --model_dir "$AF3_MODEL_DIR"
