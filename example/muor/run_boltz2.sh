#!/bin/bash
# μ-Opioid Receptor (GPCR) — Boltz-2 β-grid sampling
# References: 4DKL (inactive, antagonist-bound), 5C1M (active, agonist-bound)
# The MSA is fetched from the ColabFold server at runtime; no a3m is committed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prs predict \
    --model boltz2 \
    --input "$SCRIPT_DIR/boltz2_input.yaml" \
    --output "$SCRIPT_DIR/output_boltz2" \
    --beta "-0.6,-0.3,0,0.3,0.6" \
    --seed 42 \
    --samples 5 \
    --use_msa_server
