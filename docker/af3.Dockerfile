# Pair representation scaling CLI on top of an AlphaFold 3 image.
#
# Build the AlphaFold 3 image from a patched checkout first:
#   prs patch-af3 /path/to/alphafold3
#   cd /path/to/alphafold3 && docker build -t alphafold3-prs -f docker/Dockerfile .
#
# Then, from the repository root:
#   docker build -f docker/af3.Dockerfile --build-arg BASE=alphafold3-prs -t prs-af3 .
#
# Run (needs the AlphaFold 3 model parameters):
#   docker run --rm --gpus all -v $PWD:/work -w /work -v /path/to/af3-weights:/weights \
#     prs-af3 prs predict --model af3 --input example/rfah/af3_input.json \
#     --output example/rfah/output_af3 --beta "-0.45,0,0.45" --model_dir /weights

ARG BASE=alphafold3
FROM ${BASE}

COPY src /opt/prs/src
COPY patches /opt/prs/patches
COPY pyproject.toml README.md LICENSE /opt/prs/

# python3 resolves to the AlphaFold 3 virtual environment through PATH.
RUN uv pip install --python "$(command -v python3)" -e /opt/prs

ENV AF3_REPO=/app/alphafold
