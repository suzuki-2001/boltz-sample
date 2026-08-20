# Docker

Two containers, one per backend. Both expose the same `prs` CLI. Pick the one
that matches the model weights you have access to.

## Boltz-2

Boltz-2 comes from PyPI and the weights are downloaded on first use, so the
image builds in one step from the repository root.

```bash
docker build -f docker/boltz2.Dockerfile -t prs-boltz2 .

docker run --rm --gpus all -v $PWD:/work -w /work prs-boltz2 \
    prs predict --model boltz2 \
                --input example/rfah/boltz2_input.yaml \
                --output example/rfah/output_boltz2 \
                --beta "-0.6,-0.3,0,0.3,0.6"
```

The weights are cached under `/cache/boltz`. Mount a volume there to keep them
across container runs.

## AlphaFold 3

AlphaFold 3 inference requires the official model parameters. Request access at
<https://github.com/google-deepmind/alphafold3>. Build the AlphaFold 3 image
from a checkout with the patch applied, then add the `prs` CLI on top.

```bash
prs patch-af3 /path/to/alphafold3

cd /path/to/alphafold3
docker build -t alphafold3-prs -f docker/Dockerfile .

cd /path/to/pair-representation-scaling
docker build -f docker/af3.Dockerfile --build-arg BASE=alphafold3-prs -t prs-af3 .

docker run --rm --gpus all \
    -v $PWD:/work -w /work \
    -v /path/to/af3-weights:/weights \
    prs-af3 \
    prs predict --model af3 \
                --input example/rfah/af3_input.json \
                --output example/rfah/output_af3 \
                --beta "-0.45,0,0.45" \
                --model_dir /weights
```

The `BASE` build argument names the AlphaFold 3 image to build on. The image
sets `AF3_REPO=/app/alphafold`, so `prs predict --model af3` finds
`run_alphafold.py` without `--af3_run`.
