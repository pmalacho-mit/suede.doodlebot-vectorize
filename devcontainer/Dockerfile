FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# binaryen -> wasm-opt (used by build_wasm.sh when present); build deps for the Python wheel.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      binaryen build-essential pkg-config libssl-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
