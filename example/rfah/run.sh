#!/bin/bash
# Run boltz-sample for RfaH (fold-switching protein)
# Reference: 2OUG_C (alpha-helix CTD), 6C6S_D (beta-barrel CTD)
# Usage: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

boltz sample \
    --fasta "$SCRIPT_DIR/data/rfah.fasta" \
    --msa "$SCRIPT_DIR/data/rfah.a3m" \
    --out_dir "$SCRIPT_DIR/output" \
    --diffusion_samples 10 \
    --seed 42

boltz evaluate "$SCRIPT_DIR/output" \
    --ref "$SCRIPT_DIR/data/ref/2OUG_C.pdb" \
    --ref "$SCRIPT_DIR/data/ref/6C6S_D.pdb"
