#!/bin/bash
# RfaH (fold-switching protein) — AlphaFold 3 β-grid sampling
# References: 2OUG_C (α-helix CTD), 6C6S_D (β-barrel CTD)
# Requires: AlphaFold 3 model weights (request from Google) at $AF3_MODEL_DIR

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to your AlphaFold 3 model parameters directory}"

prs predict \
    --model af3 \
    --input "$SCRIPT_DIR/af3_input.json" \
    --output "$SCRIPT_DIR/output_af3" \
    --beta "-0.6,-0.3,0,0.3,0.6" \
    --samples 5 \
    --model_dir "$AF3_MODEL_DIR"
