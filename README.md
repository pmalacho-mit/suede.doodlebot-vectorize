> [!NOTE]
> This is a [suede](https://github.com/pmalacho-mit/suede) dependency. 

# arcline_vectorize — Rust / WebAssembly port of `arc_line_vectorization_suede`

A stage-for-stage port of the Python `default_pipeline` (image → skeleton →
segments → stroke graph → arc/line fit → joint solve → beautify → Eulerian
routing → firmware-time route optimisation → robot commands + SVG) to Rust,
compiled to WebAssembly.

| | Python (reference) | Rust native | wasm (Node 22 / Chromium) |
|---|---|---|---|
| mean over the 22 test images (1024×1024) | **6.8 s** (0.9 – 23.9 s) | **146 ms** (87 – 284 ms) | **≈190 ms** (75 – 435 ms) |

Output of the wasm module is bit-identical to the native binary on all 22
images.

This folder is the complete source: Rust crate (`Cargo.toml`, `src/`), the `vectorize` CLI, the Python
extension (`pyproject.toml`, `python/vectorize_rust.py`) and the wasm build (`build_wasm.sh`). All commands
below run from this folder.

## Prebuilt builds

Every [GitHub release](https://github.com/pmalacho-mit/suede.doodlebot-vectorize/releases) carries
the compiled module, so you can install it by URL without a Rust toolchain (replace `<ver>` with the version):

```
# Node.js (CommonJS)
npm install https://github.com/pmalacho-mit/suede.doodlebot-vectorize/releases/download/v<ver>/arcline-vectorize-node-<ver>.tgz
# Browser / bundler (ES module)
npm install https://github.com/pmalacho-mit/suede.doodlebot-vectorize/releases/download/v<ver>/arcline-vectorize-web-<ver>.tgz
# Python >= 3.9 (pip picks the wheel for your platform: Linux x86_64/aarch64, macOS universal2, Windows x64)
pip install arcline-vectorize --find-links https://github.com/pmalacho-mit/suede.doodlebot-vectorize/releases/expanded_assets/v<ver>
```

The npm packages are `arcline-vectorize-node` / `arcline-vectorize-web`: import from the package name
instead of `./pkg-node/…` / `./pkg-web/…` in the examples below.

## Using it from JavaScript

The module takes **raw RGBA8 pixels** (exactly what `CanvasRenderingContext2D.getImageData()` gives you)
and returns the same JSON structure the Python endpoint returns.

```js
import init, { vectorize_json, vectorize } from './pkg-web/arcline_vectorize.js';
await init();                                   // loads arcline_vectorize_bg.wasm (same directory)

const ctx = canvas.getContext('2d', { willReadFrequently: true });
const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);

// treat_alpha=true composites transparent pixels over white (what skimage's rgba2rgb does);
// for opaque images it makes no difference, so `true` is the safe default from a canvas.
const result = vectorize(data, width, height, true);   // JS object
// or: JSON.parse(vectorize_json(data, width, height, true))

result.low_geometry   // [{kind:'spin', degrees}, {kind:'line', distance, penDown}, {kind:'arc', radius, degrees}, ...]
result.high_geometry  // same shape
result.svg            // side-by-side comparison SVG (commands_to_svg_compare)
result.low_geometry_svg
```

`treat_alpha=false` reproduces the Python behaviour for RGB PNGs (alpha is ignored).
Images of any size work; the test set is 1024×1024.

The page must be served over HTTP (not `file://`) because the glue fetches the `.wasm`.

The module is single-threaded and has no dependencies at runtime. For UI work, call it from a Web Worker
so the ~100–400 ms of compute doesn't block rendering. The `.wasm` is ~700 KB uncompressed (≈250 KB gzipped).

## Using it from Python / FastAPI

The crate also builds as a CPython extension (PyO3, abi3 ≥ 3.9) with `maturin` (or use a
[prebuilt wheel](#prebuilt-builds)):

```
pip install maturin
maturin build --release --out dist   # -> dist/arcline_vectorize-*.whl, or `maturin develop --release`
pip install dist/arcline_vectorize-*.whl
```

```python
import arcline_vectorize
result = arcline_vectorize.vectorize(rgba_bytes, width, height, has_alpha)   # same dict as the HTTP endpoint
```

`python/vectorize_rust.py` is a drop-in `run_vectorization(image_array)` with the same contract as the one in
`vectorize.py`, so the FastAPI endpoint stays as it is:

```python
from .vectorize_rust import run_vectorization   # instead of the Python pipeline
...
return await asyncio.to_thread(run_vectorization, image_array)
```

The extension releases the GIL while it runs, so `asyncio.to_thread` (or a thread pool) gives true
parallelism across concurrent requests. Output is identical to the CLI and wasm builds (verified on the 22 images).

## Using it from Rust / the command line

```
cargo build --release --features cli
./target/release/vectorize input.png out.json            # same JSON as the wasm API
./target/release/vectorize input.png out.json --time     # prints the total time
VEC_TIMING=1 ./target/release/vectorize input.png out.json   # per-stage timings on stderr
./target/release/vectorize input.png out.json --stages dir/  # dumps intermediate stages (binary, skeleton, polylines, primitives…)
```

Library API: `arcline_vectorize::vectorize_rgba(&rgba, width, height, has_alpha) -> VectorizationResult`
(serde-serialisable), or `run_pipeline(...)` which also returns the intermediate `StageOutputs`.

## Building the wasm module yourself

Or use a [prebuilt package](#prebuilt-builds).

```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129    # must match wasm-bindgen in Cargo.lock
./build_wasm.sh
```

Output goes to `pkg-web/` (ES module for browsers) and `pkg-node/` (CommonJS for Node). `wasm-opt -O3`
(Binaryen ≥ 117) is applied when it is on the `PATH`; older versions are skipped because they break the module.

`build_wasm.sh` enables the `simd128` target feature (supported by every browser since 2021 and by Node ≥16).
SIMD only changes speed, never results: all floating-point math in the port is IEEE-exact or explicitly
emulated, so native, wasm and wasm+SIMD builds produce identical bits.
