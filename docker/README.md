# Docker

Two containers are provided, one per backend. Both expose the same `prs`
CLI; pick the backend that matches the model weights you have access to.

## Boltz-2

```bash
docker build -f third_party/boltz/docker/Dockerfile -t prs-boltz2 .

docker run --rm --gpus all -v $PWD:/work -w /work prs-boltz2 \
    prs predict --model boltz2 \
                --input example/rfah/boltz2_input.yaml \
                --output example/rfah/output_boltz2 \
                --beta "-0.6,-0.3,0,0.3,0.6" \
                --use_msa_server
```

The Boltz-2 weights are downloaded on first use and cached under
`/cache/boltz` (mount a volume there to persist them across container runs).

## AlphaFold 3

AlphaFold 3 inference requires the official model parameters; request access
at <https://github.com/google-deepmind/alphafold3>. Once you have the weights
locally:

```bash
# Build from the repository root (Dockerfile expects the prs CLI source too).
docker build -f third_party/alphafold3/docker/Dockerfile -t prs-af3 .

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

The PRS β-uniform patch is already applied to the vendored AF3 source under
`third_party/alphafold3/`. The Dockerfile additionally installs the `prs`
CLI wrapper so a β-grid sweep is a single command inside the container.
