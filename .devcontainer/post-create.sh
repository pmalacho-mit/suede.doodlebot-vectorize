#!/usr/bin/env bash
# Runs once after the container is created. Everything here is also fine to run by hand.
set -euo pipefail

# Rust via rustup (not the devcontainer feature: that feature requests
# seccomp=unconfined / SYS_PTRACE for debugging, which hardened runtimes refuse).
if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --component rust-src
fi
export PATH="$HOME/.cargo/bin:$PATH"
grep -q '.cargo/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"

rustup target add wasm32-unknown-unknown

# wasm-bindgen-cli must match the wasm-bindgen crate version in Cargo.lock.
WBG_VER=$(grep -A1 '^name = "wasm-bindgen"$' release/Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')
cargo install wasm-bindgen-cli --version "$WBG_VER" --locked

# Python: maturin builds the extension wheel; Pillow/numpy are for the helper and reference scripts.
pip install --user maturin pillow numpy

# Optional: wasm-opt shrinks the .wasm a little. build_wasm.sh skips it when it's missing.
# Not apt's binaryen: that one (v108) corrupts wasm-bindgen's externref table.
BINARYEN=version_123
( mkdir -p "$HOME/.local" \
  && curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/$BINARYEN/binaryen-$BINARYEN-$(uname -m)-linux.tar.gz" \
     | tar xz -C "$HOME/.local" \
  && mkdir -p "$HOME/.local/bin" && ln -sf "$HOME/.local/binaryen-$BINARYEN/bin/wasm-opt" "$HOME/.local/bin/wasm-opt" ) \
  || echo "binaryen not installed (optional)"

echo
echo "Ready. Useful commands:"
echo "  cargo build --manifest-path release/Cargo.toml --release --features cli"
echo "  ./release/target/release/vectorize image.png out.json --time"
echo "  ./release/build_wasm.sh             # wasm + JS glue -> release/pkg-web/ release/pkg-node/"
echo "  maturin develop --release -m release/Cargo.toml  # install the Python extension into the active interpreter"
echo "  python -m http.server 8000          # then open http://localhost:8000/web/"