//! Image loading / binarization and the stroke-width-scaled configuration tree
//! (`auto_config.py`).

use crate::ndimage::{Grid, Mask};
use crate::numeric::np_median;

/// Convert an RGBA8 buffer (row-major, 4 bytes per pixel) to the float
/// grayscale skimage produces: uint8 -> `img * (1/255)`, RGBA composited over
/// white (`rgba2rgb`), then `rgb @ [0.2125, 0.7154, 0.0721]`.
pub fn rgba_to_gray(rgba: &[u8], w: usize, h: usize, has_alpha: bool) -> Grid<f64> {
    let inv = 1.0 / 255.0;
    let mut out = Grid::new(h, w, 0.0f64);
    for i in 0..w * h {
        let r = rgba[4 * i] as f64 * inv;
        let g = rgba[4 * i + 1] as f64 * inv;
        let b = rgba[4 * i + 2] as f64 * inv;
        let (r, g, b) = if has_alpha {
            let a = rgba[4 * i + 3] as f64 * inv;
            // rgba2rgb with white background: (1 - a) * 1.0 + a * c, clipped to [0, 1]
            let comp = |c: f64| ((1.0 - a) * 1.0 + a * c).clamp(0.0, 1.0);
            (comp(r), comp(g), comp(b))
        } else {
            (r, g, b)
        };
        out.data[i] = r * 0.2125 + g * 0.7154 + b * 0.0721;
    }
    out
}

/// `binarize(rgba_to_gray(...), threshold)` without materialising the gray image.
pub fn binarize_rgba(rgba: &[u8], w: usize, h: usize, has_alpha: bool, threshold: f64) -> Mask {
    let inv = 1.0 / 255.0;
    let mut m = Mask::new(h, w, false);
    for i in 0..w * h {
        let r = rgba[4 * i] as f64 * inv;
        let g = rgba[4 * i + 1] as f64 * inv;
        let b = rgba[4 * i + 2] as f64 * inv;
        let (r, g, b) = if has_alpha {
            let a = rgba[4 * i + 3] as f64 * inv;
            let comp = |c: f64| ((1.0 - a) * 1.0 + a * c).clamp(0.0, 1.0);
            (comp(r), comp(g), comp(b))
        } else {
            (r, g, b)
        };
        m.data[i] = r * 0.2125 + g * 0.7154 + b * 0.0721 < threshold;
    }
    m
}

pub fn binarize(gray: &Grid<f64>, threshold: f64) -> Mask {
    let mut m = Mask::new(gray.h, gray.w, false);
    for i in 0..gray.data.len() {
        m.data[i] = gray.data[i] < threshold;
    }
    m
}

pub const REF_STROKE_WIDTH_PX: f64 = 3.0;

pub fn estimate_stroke_width(binary: &Mask, dt: &Grid<f64>) -> f64 {
    if !binary.any() {
        return REF_STROKE_WIDTH_PX;
    }
    let vals: Vec<f64> = (0..dt.data.len()).filter(|&i| binary.data[i]).map(|i| dt.data[i]).collect();
    if vals.is_empty() {
        return REF_STROKE_WIDTH_PX;
    }
    2.0 * np_median(&vals)
}

fn round_int(value: f64, lo: i64) -> i64 {
    lo.max(crate::numeric::py_round(value) as i64)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub stroke_width_px: f64,
    pub scale: f64,
    // collapse
    pub max_hole_area: i64,
    pub max_thin_thickness: f64,
    // eyes
    pub eyes_max_skeleton_pixels: i64,
    pub eyes_min_fill_ratio: f64,
    pub eyes_min_area: i64,
    // detect (unscaled)
    pub local_tau_radius: usize,
    pub fat_ratio: f64,
    pub min_fat_area: usize,
    pub group_dilate: usize,
    pub skel_ring_dilate: usize,
    pub pairing_tangent_steps: usize,
    pub pairing_threshold: f64,
    pub min_chromosome_skel_length: usize,
    // segment
    pub seg_min_length: f64,
    // fuse
    pub fuse_max_path_length: i64,
    pub fuse_lookback: usize,
    pub fuse_min_tangent_score: f64,
    pub fuse_gap_penalty: f64,
    pub fuse_curvature_penalty: f64,
    // repair
    pub rep_junction_tol: f64,
    pub rep_stable_skip: i64,
    pub rep_stable_sample: i64,
    pub rep_max_junction_region_length: i64,
    pub rep_min_output_polyline_length: usize,
    pub rep_min_tangent_spread_deg: f64,
    pub rep_interp_max_spacing: f64,
    pub rep_min_curvature_spike_ratio: f64,
    pub rep_curvature_context_window: i64,
    pub rep_cascade_gap: i64,
    pub rep_cascade_max_jp_distance: f64,
    // post repair fuse
    pub prf_junction_tol: f64,
    pub prf_tangent_skip: usize,
    pub prf_tangent_sample: usize,
    pub prf_min_tangent_score: f64,
    pub prf_curvature_penalty: f64,
    // graph build
    pub gb_junction_tol: f64,
    pub gb_terminal_tangent_window: usize,
    pub gb_crossing_tangent_skip: i64,
    pub gb_crossing_tangent_half_window: i64,
    pub gb_cusp_angle_threshold_deg: f64,
    pub gb_cluster_merge_centroid_distance: f64,
    pub gb_cluster_merge_index_gap: i64,
    // high geometry
    pub hg_sigma: f64,
    pub hg_corner_threshold: f64,
    pub hg_max_fit_residual: f64,
    // optimize route
    pub opt_pixels_per_inch: f64,
    pub opt_pen_up_join_tol: f64,
    pub opt_two_opt_passes: usize,
    pub opt_or_opt_passes: usize,
    pub opt_or_opt_max_segment_len: usize,
}

pub fn derive_configs(binary: &Mask, dt: &Grid<f64>) -> Config {
    let sw = estimate_stroke_width(binary, dt);
    let s = sw / REF_STROKE_WIDTH_PX;
    let s2 = s * s;
    Config {
        stroke_width_px: sw,
        scale: s,
        max_hole_area: round_int(10.0 * s2, 1),
        max_thin_thickness: 3.0 * s,
        eyes_max_skeleton_pixels: round_int(8.0 * s, 1),
        eyes_min_fill_ratio: 4.0,
        eyes_min_area: round_int(9.0 * s2, 1),
        local_tau_radius: 40,
        fat_ratio: 1.3,
        min_fat_area: 8,
        group_dilate: 15,
        skel_ring_dilate: 5,
        pairing_tangent_steps: 8,
        pairing_threshold: 1.2,
        min_chromosome_skel_length: 15,
        seg_min_length: 10.0 * s,
        fuse_max_path_length: round_int(20.0 * s, 2),
        fuse_lookback: 10,
        fuse_min_tangent_score: 0.5,
        fuse_gap_penalty: 0.05,
        fuse_curvature_penalty: 3.0,
        rep_junction_tol: 2.5 * s,
        rep_stable_skip: 2,
        rep_stable_sample: 6,
        rep_max_junction_region_length: round_int(20.0 * s, 2),
        rep_min_output_polyline_length: 2,
        rep_min_tangent_spread_deg: 15.0,
        rep_interp_max_spacing: 1.0 * s,
        rep_min_curvature_spike_ratio: 2.0,
        rep_curvature_context_window: 8,
        rep_cascade_gap: 3,
        rep_cascade_max_jp_distance: 3.0 * s,
        prf_junction_tol: 2.5 * s,
        prf_tangent_skip: 2,
        prf_tangent_sample: 10,
        prf_min_tangent_score: 0.6,
        prf_curvature_penalty: 1.0,
        gb_junction_tol: 2.5 * s,
        gb_terminal_tangent_window: 10,
        gb_crossing_tangent_skip: 2,
        gb_crossing_tangent_half_window: 6,
        gb_cusp_angle_threshold_deg: 50.0,
        gb_cluster_merge_centroid_distance: 10.0 * s,
        gb_cluster_merge_index_gap: 10,
        hg_sigma: 2.0,
        hg_corner_threshold: 0.25,
        hg_max_fit_residual: 5.0 * s,
        opt_pixels_per_inch: 1.0,
        opt_pen_up_join_tol: 0.5 * s,
        opt_two_opt_passes: 16,
        opt_or_opt_passes: 8,
        opt_or_opt_max_segment_len: 3,
    }
}
