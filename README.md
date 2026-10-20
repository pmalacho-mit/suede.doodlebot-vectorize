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


This repository is a [suede](https://github.com/pmalacho-mit/suede) dependency. Everything a consumer
needs to build the module lives in [`release/`](release/) (which is published to the `release` branch);
see [release/README.md](release/README.md) for how to use it from JavaScript, Python and Rust.
Everything outside `release/` is development, verification and CI.

## Layout

```
release/                  shipped source (the `release` branch)
  Cargo.toml / Cargo.lock crate `arcline_vectorize` (rlib + cdylib) + `vectorize` CLI (feature `cli`) + Python ext (feature `python`)
  src/                    the port (module map below)
  pyproject.toml          maturin config for the Python wheel (version comes from Cargo.toml)
  python/vectorize_rust.py  drop-in `run_vectorization` for the FastAPI app
  build_wasm.sh           builds the wasm module and generates JS glue into release/pkg-web/ and release/pkg-node/ (gitignored)
  README.md               consumer documentation
web/index.html            browser demo (file picker → canvas → wasm → commands + SVG)
.devcontainer/            development container
.github/workflows/        CI: build-release.yml (below) + suede's publish workflow
js/bench_node.js          Node benchmark / parity check against native JSON outputs
tests/reference/          Python reference outputs, Rust outputs and the parity report for the 22 images
```

## Development container

`.devcontainer/` sets up Ubuntu 24.04 with Rust (stable + `wasm32-unknown-unknown`), Python 3.12, Node 22,
`wasm-bindgen-cli` pinned to the version in `release/Cargo.lock`, `maturin` and Binaryen 123.

```
cargo build --manifest-path release/Cargo.toml --release --features cli
./release/build_wasm.sh
maturin develop --release -m release/Cargo.toml
python3 -m http.server 8000      # then open http://localhost:8000/web/ (the demo loads release/pkg-web/)
```

## Releases

[`.github/workflows/build-release.yml`](.github/workflows/build-release.yml) builds from `release/` and attaches
to a GitHub release: the Node and web wasm packages (`npm install <url>`), abi3 wheels for Linux x86_64/aarch64,
macOS universal2 and Windows x64, and an sdist. Each build is smoke-tested before publishing.

1. Bump `version` in `release/Cargo.toml` (the wheel takes its version from there).
2. Push a tag `v<version>` (the workflow refuses a tag that doesn't match), or run the workflow by hand
   from the Actions tab, which tags the current commit as `v<version>`.

Pull requests touching `release/` run the same builds without publishing. Built artifacts never go into
`release/`; it holds source only.

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
cargo build --manifest-path release/Cargo.toml --release --features cli
for f in images/*.png; do ./release/target/release/vectorize "$f" tests/reference/rust/$(basename "$f" .png).json; done
# wasm vs native
node js/bench_node.js <dir with .rgba dumps + meta.json> tests/reference/rust
```
