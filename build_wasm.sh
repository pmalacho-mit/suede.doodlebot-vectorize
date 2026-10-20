#!/bin/bash
# Builds the wasm module and generates JS glue for Node (pkg-node/) and the browser (pkg-web/).
# Requires: wasm32-unknown-unknown target (or rust-src for -Zbuild-std) and wasm-bindgen-cli
# matching the wasm-bindgen version in Cargo.lock.
set -e
cd "$(dirname "$0")"
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --features wasm
else
  RUSTFLAGS="-C target-feature=+simd128" RUSTC_BOOTSTRAP=1 cargo build --release --target wasm32-unknown-unknown --features wasm -Zbuild-std=std,panic_abort
fi
WASM=target/wasm32-unknown-unknown/release/arcline_vectorize.wasm
wasm-bindgen "$WASM" --out-dir pkg-node --target nodejs
wasm-bindgen "$WASM" --out-dir pkg-web --target web
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 pkg-web/arcline_vectorize_bg.wasm -o pkg-web/arcline_vectorize_bg.wasm
  wasm-opt -O3 pkg-node/arcline_vectorize_bg.wasm -o pkg-node/arcline_vectorize_bg.wasm
fi
ls -la pkg-web/*.wasm
