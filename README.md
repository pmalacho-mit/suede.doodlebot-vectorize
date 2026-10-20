# arcline_vectorize — Rust / WebAssembly port of `arc_line_vectorization_suede`

A stage-for-stage port of the Python `default_pipeline` (image → skeleton →
segments → stroke graph → arc/line fit → joint solve → beautify → Eulerian
routing → firmware-time route optimisation → robot commands + SVG) to Rust,
compiled to WebAssembly.

| | Python (reference) | Rust native | wasm (Node 22 / Chromium) |
|---|---|---|---|
| mean over the 22 test images (1024×1024) | **6.8 s** (0.9 – 23.9 s) | **146 ms** (87 – 284 ms) | **≈190 ms** (75 – 435 ms) |

Output of the wasm module is bit-identical to the native binary on all 22
images. Parity against the Python reference is described in detail below.

## Layout

```
Cargo.toml            crate `arcline_vectorize` (rlib + cdylib) + `vectorize` CLI (feature `cli`) + Python ext (feature `python`)
pyproject.toml        maturin config for the Python wheel
python/vectorize_rust.py  drop-in `run_vectorization` for the FastAPI app
src/                  the port (module map below)
build_wasm.sh         builds the wasm module and generates JS glue into pkg-web/ and pkg-node/
pkg-web/              prebuilt ES-module build for browsers   (arcline_vectorize.js + _bg.wasm, ~700 KB)
pkg-node/             prebuilt CommonJS build for Node
web/index.html        browser demo (file picker → canvas → wasm → commands + SVG)
js/bench_node.js      Node benchmark / parity check against native JSON outputs
tests/reference/      Python reference outputs, Rust outputs and the parity report for the 22 images
```

## Development container

`.devcontainer/` sets up Ubuntu 24.04 with Rust (stable + `wasm32-unknown-unknown`), Python 3.12, Node 22,
`wasm-bindgen-cli` pinned to the version in `Cargo.lock`, `maturin` and `binaryen`. The post-create step also
runs a native build and `./build_wasm.sh`, so the container opens ready to use.

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

The page must be served over HTTP (not `file://`) because the glue fetches the `.wasm`:

```
python3 -m http.server 8000      # then open http://localhost:8000/web/
```

The module is single-threaded and has no dependencies at runtime. For UI work, call it from a Web Worker
so the ~100–400 ms of compute doesn't block rendering. The `.wasm` is ~700 KB uncompressed (≈250 KB gzipped);
`wasm-opt -O3` is applied by `build_wasm.sh` when Binaryen is installed.

## Using it from Python / FastAPI

The crate also builds as a CPython extension (PyO3, abi3 ≥ 3.9) with `maturin`:

```
pip install maturin
maturin build --release            # -> target/wheels/arcline_vectorize-*.whl, or `maturin develop --release`
pip install target/wheels/arcline_vectorize-*.whl
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

```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129    # must match wasm-bindgen in Cargo.lock
./build_wasm.sh
```

`build_wasm.sh` enables the `simd128` target feature (supported by every browser since 2021 and by Node ≥16).
SIMD only changes speed, never results: all floating-point math in the port is IEEE-exact or explicitly
emulated, so native, wasm and wasm+SIMD builds produce identical bits.

## Parity with the Python reference

Reference outputs were produced with the uploaded `code/` package on CPython 3.12, numpy 2.x (OpenBLAS),
scipy, scikit-image, networkx on x86-64 Linux (`tests/reference/python/`). Rust outputs for the same
inputs are in `tests/reference/rust/`; `tests/reference/parity_report.md` is the full table.

### What is bit-identical

* binarisation, stroke-width estimation and the whole derived configuration tree
* skeletonisation (Zhang-Suen, hole collapse with Lee re-thinning, filled-region "eyes",
  ribbon-collapse crossing detection and resolution) — the skeleton masks are identical pixel for pixel
* segmentation, A\* fusion through the binary and junction repair — raw segments identical
* polylines agree to ~1e-13 (sum-order effects only), stroke graph identical
* **high-geometry commands: identical sequence, values agree to ≤ 7e-12 relative** on all 22 images
* the SVG strings differ only where the numbers above differ (they print with 2 decimals, so
  high-geometry SVGs match; low-geometry SVGs differ in a few digits)

### Low geometry — the honest picture

The low-geometry path runs two `scipy.optimize.least_squares(method='trf', tr_solver='lsmr')`
joint solves per image with a finite-difference Jacobian and `max_nfev=80`. On every test image
both solves **stop at the iteration cap, unconverged**. That makes the exact result a function of the
whole floating-point trajectory, and the trajectory diverges from the reference at the level of
LAPACK internals that are not reproducible across machines (`dgelsd` inside the Kasa circle fit,
x87-based `dnrm2`, platform BLAS kernels — your Mac's Accelerate differs from the Linux OpenBLAS the
reference ran on in exactly the same way). The port reproduces numpy's BLAS kernel behaviour
(FMA-fused `ddot`/`dgemv` accumulation orders, pairwise summation, LAPACK sign conventions, MINPACK
and TRF/LSMR step logic, networkx insertion-order and CPython set-iteration semantics), and the
residual function evaluates bit-identically to numpy — but the 1-ulp differences that remain in the
finite-difference Jacobian are amplified by ~1e8 and the unconverged solver then stops at a slightly
different point.

Concretely, on the 22 images:

* **21/22 images: identical low-geometry command sequence** (same kinds, same pen states, same count).
  Values agree to within 1e-6 relative for 7–69 % of the numbers and to within **≤ 0.05° / ≤ 0.3 % of radius**
  for the rest. Simulating both command lists and comparing the drawn paths gives a Hausdorff distance
  of **≤ 0.1 px on 19 images, 0.5 px on 2 images** (1024-px canvas).
* **1/22 (`combined_20260716_201214_137375`)**: 136 vs 138 commands. One arc fitted slightly differently
  after the unconverged solve, and the route optimiser then chose a different order; the drawn path
  differs by at most 1.07 px.

If exact numerical agreement on the low-geometry path matters more than speed, the pragmatic options
are (a) raise `max_nfev` in both implementations until the solves converge (then both land on the same
minimum), or (b) treat the Python output as the acceptance test with a geometric tolerance (the
`tests/reference` scripts do this). Neither changes the port.

### Things that were intentionally not ported

They do not affect `low_geometry`, `high_geometry` or the SVGs:
skeleton/segment/command *labels* (watershed labelling, KD-tree nearest-segment lookup), `commands_fitted`,
the DP chain subdivision helper, and the Python debugging visualisations. The CLI's `--stages` dump covers
what is needed for parity testing instead.

One behavioural difference: on an image whose segmentation yields an empty chain the Python reference
raises `IndexError` (e.g. `noise.png`); the port skips that segment and continues.

## Module map (Python → Rust)

| Python | Rust | Notes |
|---|---|---|
| `auto_config.py`, image loading | `config.rs` | skimage gray coefficients, `stroke_width = 2·median(EDT)` |
| scipy.ndimage / skimage helpers | `ndimage.rs`, `skeleton.rs` | label, exact EDT, uniform filter (running sums), gaussian_filter1d, bresenham, find_peaks, Zhang-Suen, Lee |
| `skeletonize*.py`, `crossings.py` | `skel_stage.rs` | hole collapse, eyes, ribbon-collapse crossings |
| `segment*.py`, `repair*.py` | `segment.rs` | tracing, A\* fusion, junction repair, post-repair fusion |
| `stroke_graph.py` | `graph.rs` | |
| `fit*.py`, circle/line fits | `lowgeo/fitting.rs`, `lowgeo/minpack.rs` | top-down fit, Kasa + MINPACK `lmder`, LAPACK-faithful SVD |
| joint solve (`least_squares` TRF + lsmr) | `lowgeo/solve.rs` | sparse FD Jacobian, LSMR, 2-D trust region |
| beautify / merge arc pairs | `lowgeo/beautify.rs` | |
| networkx usage (eulerize, matching, routing) | `lowgeo/nxgraph.rs`, `lowgeo/routing.rs` | MultiGraph insertion order, blossom matching, CPython set-of-tuples order |
| numpy / BLAS numerics | `numeric.rs`, `blas.rs`, `fmath.rs` | pairwise sums, OpenBLAS `ddot` emulation with correctly-rounded software FMA on wasm, `libm` for transcendentals |
| high geometry | `highgeo.rs` | |
| `optimize_route` | `optimize.rs` | NN + 2-opt + or-opt on the firmware time model |
| `visualize.commands_to_svg*` | `svg.rs` | |
| `commands` | `commands.rs` | `{kind, distance, penDown}` / `{kind, degrees}` / `{kind, radius, degrees}` |

## Reproducing the parity report

```
# Python reference (needs the original package + numpy/scipy/scikit-image/networkx/Pillow)
python tests/reference/gen_ref.py <path-to-code> tests/reference/python 'images/*.png'
# Rust
cargo build --release --features cli
for f in images/*.png; do ./target/release/vectorize "$f" tests/reference/rust/$(basename "$f" .png).json; done
# wasm vs native
node js/bench_node.js <dir with .rgba dumps + meta.json> tests/reference/rust
```
