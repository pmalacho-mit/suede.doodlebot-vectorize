//! Rust port of the arc/line vectorization pipeline: turns a raster drawing
//! into robot drawing commands (`line` / `spin` / `arc`), matching the Python
//! reference implementation's `default_pipeline` output.

pub mod fmath;
pub mod blas;
pub mod commands;
pub mod config;
pub mod graph;
pub mod highgeo;
pub mod lowgeo;
pub mod ndimage;
pub mod numeric;
pub mod optimize;
pub mod segment;
pub mod skel_stage;
pub mod skeleton;
pub mod svg;

use commands::Command;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct VectorizationResult {
    pub low_geometry: Vec<Command>,
    pub high_geometry: Vec<Command>,
    pub svg: String,
    pub low_geometry_svg: String,
}

/// Intermediate stage outputs (for debugging / parity testing).
pub struct StageOutputs {
    pub binary: ndimage::Mask,
    pub skeleton: ndimage::Mask,
    pub raw_segments: Vec<segment::Polyline>,
    pub polylines: Vec<segment::Polyline>,
    pub low_unoptimized: Vec<Command>,
    pub high_unoptimized: Vec<Command>,
    pub prims_initial: Vec<lowgeo::primitives::Prim>,
    pub prims_fitted: Vec<lowgeo::primitives::Prim>,
    pub prims_consolidated: Vec<lowgeo::primitives::Prim>,
    pub tour: Vec<(usize, bool)>,
}

/// Run the full pipeline on an RGBA8 image buffer (row-major, 4 bytes/pixel).
/// `has_alpha=false` treats the buffer as opaque RGB(A), exactly like the
/// Python pipeline does for RGB PNGs.
pub fn run_pipeline(rgba: &[u8], width: usize, height: usize, has_alpha: bool) -> (VectorizationResult, StageOutputs) {
    let mut timer = StageTimer::new();
    let binary = config::binarize_rgba(rgba, width, height, has_alpha, 0.5);
    timer.lap("binarize");
    let dt = ndimage::distance_transform_edt(&binary);
    timer.lap("EDT");
    let cfg = config::derive_configs(&binary, &dt);
    timer.lap("derive_configs");
    let skeleton = skel_stage::skeletonize_stage(&binary, &dt, &cfg);
    timer.lap("skeletonize stage");
    let (raw_segments, polylines) = segment::segment_stage(&skeleton, &binary, &cfg);
    timer.lap("segment stage");
    let graph = graph::build_graph(polylines.clone(), &cfg);
    timer.lap("stroke graph");
    let start_pos = [0.0, 0.0];
    let start_heading = 0.0;
    let low = lowgeo::vectorize_low_geometry(&graph, start_pos, start_heading);
    timer.lap("low geometry (fit + solve + route)");
    let high = highgeo::vectorize_high_geometry(&polylines, start_pos, start_heading, &cfg);
    timer.lap("high geometry");
    let low_opt = optimize::optimize_route(&low.commands_consolidated, start_pos, start_heading, &cfg);
    let high_opt = optimize::optimize_route(&high, start_pos, start_heading, &cfg);
    timer.lap("route optimization");
    let svg = svg::commands_to_svg_compare(&low_opt, &high_opt, "low_geometry_optimized.commands", "high_geometry_optimized.commands");
    let low_svg = svg::commands_to_svg(&low_opt);
    timer.lap("svg");
    (
        VectorizationResult { low_geometry: low_opt, high_geometry: high_opt, svg, low_geometry_svg: low_svg },
        StageOutputs {
            binary,
            skeleton,
            raw_segments,
            polylines,
            low_unoptimized: low.commands_consolidated,
            high_unoptimized: high,
            prims_initial: low.prims_initial,
            prims_fitted: low.prims_fitted,
            prims_consolidated: low.prims_consolidated,
            tour: low.tour,
        },
    )
}

/// Optional per-stage timing printed to stderr when `VEC_TIMING` is set (native only).
pub(crate) struct StageTimer {
    #[cfg(not(target_arch = "wasm32"))]
    t: Option<std::time::Instant>,
}
impl StageTimer {
    pub(crate) fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            StageTimer { t: if timing_enabled() { Some(std::time::Instant::now()) } else { None } }
        }
        #[cfg(target_arch = "wasm32")]
        {
            StageTimer {}
        }
    }
    #[allow(unused_variables)]
    pub(crate) fn lap(&mut self, name: &str) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t) = self.t.as_mut() {
            eprintln!("  {:>8.2} ms  {}", t.elapsed().as_secs_f64() * 1000.0, name);
            *t = std::time::Instant::now();
        }
    }
}

/// Accumulating stopwatch for sub-stage timing; a no-op on wasm32.
pub(crate) struct Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    start: Option<std::time::Instant>,
    #[cfg(not(target_arch = "wasm32"))]
    pub total: std::time::Duration,
}
impl Stopwatch {
    pub(crate) fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Stopwatch { start: None, total: std::time::Duration::ZERO }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Stopwatch {}
        }
    }
    #[inline]
    pub(crate) fn start(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if timing_enabled() {
                self.start = Some(std::time::Instant::now());
            }
        }
    }
    #[inline]
    pub(crate) fn stop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(t) = self.start.take() {
                self.total += t.elapsed();
            }
        }
    }
    pub(crate) fn report(&self) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            format!("{:.2} ms", self.total.as_secs_f64() * 1000.0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            String::new()
        }
    }
}

pub(crate) fn timing_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("VEC_TIMING").is_ok())
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

pub fn vectorize_rgba(rgba: &[u8], width: usize, height: usize, has_alpha: bool) -> VectorizationResult {
    run_pipeline(rgba, width, height, has_alpha).0
}

#[cfg(feature = "wasm")]
mod wasm_api {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// Vectorize an RGBA8 pixel buffer (e.g. `ImageData.data`) and return the
    /// result as a JSON string: {low_geometry, high_geometry, svg, low_geometry_svg}.
    #[wasm_bindgen]
    pub fn vectorize_json(rgba: &[u8], width: u32, height: u32, treat_alpha: bool) -> String {
        let r = vectorize_rgba(rgba, width as usize, height as usize, treat_alpha);
        serde_json::to_string(&r).unwrap()
    }

    /// Same as `vectorize_json`, but returns a JS object directly.
    #[wasm_bindgen]
    pub fn vectorize(rgba: &[u8], width: u32, height: u32, treat_alpha: bool) -> JsValue {
        let r = vectorize_rgba(rgba, width as usize, height as usize, treat_alpha);
        serde_wasm_bindgen_lite::to_js(&r)
    }

    mod serde_wasm_bindgen_lite {
        use wasm_bindgen::prelude::*;
        pub fn to_js<T: serde::Serialize>(v: &T) -> JsValue {
            let s = serde_json::to_string(v).unwrap();
            js_sys_parse(&s)
        }
        #[wasm_bindgen(inline_js = "export function parse_json(s) { return JSON.parse(s); }")]
        extern "C" {
            #[wasm_bindgen(js_name = parse_json)]
            fn js_sys_parse(s: &str) -> JsValue;
        }
    }
}

/// Python extension module (feature `python`, built with maturin): `arcline_vectorize.vectorize(...)`.
#[cfg(feature = "python")]
mod py_api {
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict, PyList};

    fn cmds_to_py<'py>(py: Python<'py>, cmds: &[crate::commands::Command]) -> PyResult<Bound<'py, PyList>> {
        use crate::commands::Command;
        let list = PyList::empty(py);
        for c in cmds {
            let d = PyDict::new(py);
            match c {
                Command::Line { distance, pen_down } => {
                    d.set_item("kind", "line")?;
                    d.set_item("distance", *distance)?;
                    d.set_item("penDown", *pen_down)?;
                }
                Command::Spin { degrees } => {
                    d.set_item("kind", "spin")?;
                    d.set_item("degrees", *degrees)?;
                }
                Command::Arc { radius, degrees } => {
                    d.set_item("kind", "arc")?;
                    d.set_item("radius", *radius)?;
                    d.set_item("degrees", *degrees)?;
                }
            }
            list.append(d)?;
        }
        Ok(list)
    }

    /// vectorize(rgba: bytes, width: int, height: int, has_alpha: bool) -> dict
    ///
    /// `rgba` is a row-major RGBA8 buffer (`PIL.Image.convert("RGBA").tobytes()`).
    /// Pass `has_alpha=True` only for images that really carry alpha (mode RGBA/LA);
    /// the Python pipeline ignores alpha for RGB images. Releases the GIL while running.
    #[pyfunction]
    #[pyo3(signature = (rgba, width, height, has_alpha=false))]
    fn vectorize<'py>(py: Python<'py>, rgba: &Bound<'py, PyBytes>, width: usize, height: usize, has_alpha: bool) -> PyResult<Bound<'py, PyDict>> {
        let buf = rgba.as_bytes();
        if buf.len() != width * height * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} bytes of RGBA8 for {}x{}, got {}",
                width * height * 4,
                width,
                height,
                buf.len()
            )));
        }
        let r = py.allow_threads(|| crate::vectorize_rgba(buf, width, height, has_alpha));
        let d = PyDict::new(py);
        d.set_item("low_geometry", cmds_to_py(py, &r.low_geometry)?)?;
        d.set_item("high_geometry", cmds_to_py(py, &r.high_geometry)?)?;
        d.set_item("svg", r.svg)?;
        d.set_item("low_geometry_svg", r.low_geometry_svg)?;
        Ok(d)
    }

    #[pymodule]
    fn arcline_vectorize(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(vectorize, m)?)?;
        Ok(())
    }
}
