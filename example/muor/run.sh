#!/bin/bash
# Run boltz-sample for mu-opioid receptor
# Reference: 4DKL_7tm (inactive), 5C1M_7tm (active)
# Usage: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

boltz sample \
    --fasta "$SCRIPT_DIR/data/muor_7tm.fasta" \
    --msa "$SCRIPT_DIR/data/muor_7tm.a3m" \
    --out_dir "$SCRIPT_DIR/output" \
    --diffusion_samples 10 \
    --seed 42

boltz evaluate "$SCRIPT_DIR/output" \
    --ref "$SCRIPT_DIR/data/ref/4DKL_7tm.pdb" \
    --ref "$SCRIPT_DIR/data/ref/5C1M_7tm.pdb"
