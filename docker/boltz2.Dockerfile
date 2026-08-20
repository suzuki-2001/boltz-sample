# Boltz-2 container with the pair representation scaling CLI.
#
# Build (from the repository root):
#   docker build -f docker/boltz2.Dockerfile -t prs-boltz2 .
#
# Run (β-grid sweep on the rfah example):
#   docker run --rm --gpus all -v $PWD:/work -w /work prs-boltz2 \
#     prs predict --model boltz2 --input example/rfah/boltz2_input.yaml \
#     --output example/rfah/output_boltz2 --beta "-0.6,-0.3,0,0.3,0.6"

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3.12 python3.12-dev python3-pip git wget ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /opt/prs

COPY src /opt/prs/src
COPY patches /opt/prs/patches
COPY pyproject.toml README.md LICENSE /opt/prs/

# Ubuntu 24.04 ships an externally-managed Python (PEP 668). The container is a
# dedicated environment, so --break-system-packages is the standard approach.
# The cuda extra brings in the cuequivariance triangle kernels used on GPU.
RUN python -m pip install --no-cache-dir --break-system-packages "boltz[cuda]==2.2.1" \
    && python -m pip install --no-cache-dir --break-system-packages -e /opt/prs

ENV BOLTZ_CACHE=/cache/boltz
VOLUME ["/cache", "/work"]
WORKDIR /work
