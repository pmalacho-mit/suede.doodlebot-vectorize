#!/usr/bin/env bash
# Runs once after the container is created: toolchain targets and the two CLIs the build scripts need.
set -euo pipefail

rustup target add wasm32-unknown-unknown

# wasm-bindgen-cli must match the wasm-bindgen crate version in Cargo.lock.
WBG_VER=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')
cargo install wasm-bindgen-cli --version "$WBG_VER" --locked

# Python side: maturin for the extension wheel, Pillow/numpy for the helper + reference scripts.
pip install --user --upgrade pip
pip install --user maturin pillow numpy

# Warm the build cache and check everything links.
cargo build --release --features cli
./build_wasm.sh

echo
echo "Ready. Useful commands:"
echo "  cargo build --release --features cli && ./target/release/vectorize image.png out.json --time"
echo "  ./build_wasm.sh                     # wasm + JS glue -> pkg-web/ pkg-node/"
echo "  maturin develop --release           # install the Python extension into the active interpreter"
echo "  python -m http.server 8000          # then open http://localhost:8000/web/"
