//! Per-segment fitting and chain subdivision (`fitting.py`).

use crate::fmath::DetMath;
use super::minpack::lm_least_squares;
use super::primitives::*;
use crate::numeric::{dist, lstsq3, mean_axis0, norm2, np_max, np_min, np_sum, sub, svd_vt_n2, P2};

pub const EPS_F: f64 = 1e-9;
const MAX_RADIUS_FACTOR: f64 = 8.0;
const MIN_ARC_SAGITTA_PX: f64 = 1.0;
const SINGLE_ARC_SHORTCUT_RMS_CAP: f64 = 6.0;
const SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP: f64 = 12.0;
const SUBDIVISION_MIN_POINTS: usize = 40;

pub fn polyline_length(pts: &[P2]) -> f64 {
    if pts.len() < 2 {
        return 0.0;
    }
    let d: Vec<f64> = (1..pts.len()).map(|i| crate::numeric::dist_naive(pts[i], pts[i - 1])).collect();
    np_sum(&d)
}

pub fn is_near_closed_polyline(pts: &[P2], gap_ratio: f64, abs_tol: f64) -> bool {
    if pts.len() < 4 {
        return false;
    }
    let gap = dist(pts[0], pts[pts.len() - 1]);
    if gap < abs_tol {
        return true;
    }
    let l = polyline_length(pts);
    if l < EPS_F {
        return false;
    }
    gap / l < gap_ratio
}

pub fn find_corners(pts: &[P2]) -> Vec<usize> {
    let sharp_threshold_deg = 80.0f64;
    let soft_threshold_deg = 50.0f64;
    let sustain_indices: i64 = 3;
    let n = pts.len();
    let w = 8usize.max(30usize.min(n / 30));
    let min_separation = 12usize.max(w + 4);
    if n < 2 * w + 1 {
        return vec![];
    }
    let nt = n - 2 * w;
    let mut turns = Vec::with_capacity(nt);
    for i in 0..nt {
        let mut left = sub(pts[w + i], pts[i]);
        let mut right = sub(pts[2 * w + i], pts[w + i]);
        let ll = crate::numeric::norm2_naive(left);
        let rl = crate::numeric::norm2_naive(right);
        if ll > EPS_F && rl > EPS_F {
            left = [left[0] / ll, left[1] / ll];
            right = [right[0] / rl, right[1] / rl];
        }
        let d = crate::numeric::dot_naive(left, right).clamp(-1.0, 1.0);
        turns.push(d.lacos());
    }
    let sharp = sharp_threshold_deg.to_radians();
    let soft = soft_threshold_deg.to_radians();
    let n_turns = turns.len() as i64;
    let mut corners = Vec::new();
    let mut i: i64 = 0;
    while i < n_turns {
        let t = turns[i as usize];
        let is_sharp = t > sharp;
        let mut is_sustained = false;
        if t > soft {
            let lo = i - sustain_indices;
            let hi = i + sustain_indices;
            if lo >= 0 && hi < n_turns && turns[lo as usize] > soft && turns[hi as usize] > soft {
                is_sustained = true;
            }
        }
        if is_sharp || is_sustained {
            let j_end = n_turns.min(i + min_separation as i64);
            let mut best = i;
            let mut j = i + 1;
            while j < j_end {
                if turns[j as usize] > turns[best as usize] {
                    best = j;
                }
                j += 1;
            }
            corners.push(best as usize + w);
            i = best + min_separation as i64;
        } else {
            i += 1;
        }
    }
    corners
}

fn smoothed_tangents(pts: &[P2], smooth_window: usize) -> Option<Vec<P2>> {
    let n = pts.len();
    let diffs: Vec<P2> = (1..n).map(|i| sub(pts[i], pts[i - 1])).collect();
    if diffs.len() < smooth_window + 1 {
        return None;
    }
    let m = diffs.len() - smooth_window + 1;
    let mut sm = Vec::with_capacity(m);
    for k in 0..m {
        let v = crate::numeric::sum_axis0(&diffs[k..k + smooth_window]);
        let nrm = norm2(v);
        if nrm < EPS_F {
            sm.push([0.0, 0.0]);
        } else {
            sm.push([v[0] / nrm, v[1] / nrm]);
        }
    }
    Some(sm)
}

pub fn find_inflections(pts: &[P2]) -> Vec<usize> {
    let smooth_window = 4usize;
    let sign_threshold = 0.03;
    let min_sustain = 3usize;
    let min_separation: i64 = 12;
    let n = pts.len();
    if n < 2 * (smooth_window + min_sustain) + 2 {
        return vec![];
    }
    let sm = match smoothed_tangents(pts, smooth_window) {
        Some(s) => s,
        None => return vec![],
    };
    if sm.len() < 2 {
        return vec![];
    }
    let cross: Vec<f64> = (0..sm.len() - 1).map(|i| sm[i][0] * sm[i + 1][1] - sm[i][1] * sm[i + 1][0]).collect();
    let mut out = Vec::new();
    let mut last: i64 = -min_separation;
    let half_offset = smooth_window / 2 + 1;
    let nc = cross.len();
    let mut i = min_sustain;
    while i + min_sustain < nc {
        let left = &cross[i - min_sustain..i];
        let right = &cross[i..i + min_sustain];
        let c1 = left.iter().all(|&v| v > sign_threshold) && right.iter().all(|&v| v < -sign_threshold);
        let c2 = left.iter().all(|&v| v < -sign_threshold) && right.iter().all(|&v| v > sign_threshold);
        if c1 || c2 {
            let pos = (i + half_offset) as i64;
            if pos - last >= min_separation {
                out.push(pos as usize);
                last = pos;
            }
            i += min_separation as usize;
        } else {
            i += 1;
        }
    }
    out
}

fn has_curvature_reversal(pts: &[P2]) -> bool {
    let smooth_window = 4usize;
    let min_turns_each_sign = 2usize;
    let sign_threshold = 0.05;
    let n = pts.len();
    if n < 2 * smooth_window + 2 {
        return false;
    }
    let sm = match smoothed_tangents(pts, smooth_window) {
        Some(s) => s,
        None => return false,
    };
    if sm.len() < 2 {
        return false;
    }
    let mut pos = 0;
    let mut neg = 0;
    for i in 0..sm.len() - 1 {
        let c = sm[i][0] * sm[i + 1][1] - sm[i][1] * sm[i + 1][0];
        if c > sign_threshold {
            pos += 1;
        }
        if c < -sign_threshold {
            neg += 1;
        }
    }
    pos >= min_turns_each_sign && neg >= min_turns_each_sign
}

/// ODR line fit: returns (p0, p1, rms).
pub fn fit_line(pts: &[P2]) -> (P2, P2, f64) {
    if pts.len() < 2 {
        return (pts[0], pts[0], 0.0);
    }
    let mu = mean_axis0(pts);
    let centered: Vec<P2> = pts.iter().map(|&p| sub(p, mu)).collect();
    let vt = svd_vt_n2(&centered);
    let d = vt[0];
    let k = centered.len();
    let t: Vec<f64> = centered.iter().map(|&c| crate::blas::gemv_row(c, d, k)).collect();
    let tmin = np_min(&t);
    let tmax = np_max(&t);
    let p0 = [mu[0] + tmin * d[0], mu[1] + tmin * d[1]];
    let p1 = [mu[0] + tmax * d[0], mu[1] + tmax * d[1]];
    let sq: Vec<f64> = centered
        .iter()
        .zip(&t)
        .map(|(c, &ti)| {
            let px = c[0] - ti * d[0];
            let py = c[1] - ti * d[1];
            px * px + py * py
        })
        .collect();
    let rms = (np_sum(&sq) / sq.len() as f64).sqrt();
    (p0, p1, rms)
}

fn fit_circle_kasa(pts: &[P2]) -> (P2, f64) {
    let m: Vec<[f64; 3]> = pts.iter().map(|p| [p[0], p[1], 1.0]).collect();
    let rhs: Vec<f64> = pts.iter().map(|p| -(p[0] * p[0] + p[1] * p[1])).collect();
    let sol = match lstsq3(&m, &rhs) {
        Some(s) => s,
        None => return ([0.0, 0.0], 1e9),
    };
    let (a, b, c) = (sol[0], sol[1], sol[2]);
    let cx = -a / 2.0;
    let cy = -b / 2.0;
    let r_sq = cx * cx + cy * cy - c;
    if r_sq < 0.0 {
        return ([cx, cy], 1e9);
    }
    ([cx, cy], r_sq.sqrt())
}

fn fit_circle_geometric(pts: &[P2], c0: P2, r0: f64) -> (P2, f64, f64) {
    let resid = |params: &[f64]| -> Vec<f64> {
        pts.iter().map(|p| crate::numeric::dist_naive(*p, [params[0], params[1]]) - params[2]).collect()
    };
    let (c, r) = match lm_least_squares(resid, &[c0[0], c0[1], r0], 50) {
        Some(x) => ([x[0], x[1]], x[2]),
        None => (c0, r0),
    };
    let sq: Vec<f64> = pts
        .iter()
        .map(|p| {
            let res = crate::numeric::dist_naive(*p, c) - r;
            res * res
        })
        .collect();
    let rms = (np_sum(&sq) / sq.len() as f64).sqrt();
    (c, r, rms)
}

pub fn fit_circle(pts: &[P2]) -> (P2, f64, f64) {
    if pts.len() < 3 {
        return ([0.0, 0.0], 0.0, f64::INFINITY);
    }
    let (c0, r0) = fit_circle_kasa(pts);
    fit_circle_geometric(pts, c0, r0)
}

pub fn segment_extent(pts: &[P2]) -> f64 {
    if pts.len() < 2 {
        return 1.0;
    }
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
    let span = [np_max(&xs) - np_min(&xs), np_max(&ys) - np_min(&ys)];
    norm2(span).max(1.0)
}

/// Returns (Arc primitive or None, rms).
pub fn fit_arc(pts: &[P2]) -> (Option<Prim>, f64) {
    if pts.len() < 3 {
        return (None, f64::INFINITY);
    }
    let (c, r, rms) = fit_circle(pts);
    if !r.is_finite() || r < EPS_F {
        return (None, f64::INFINITY);
    }
    if r > MAX_RADIUS_FACTOR * segment_extent(pts) {
        return (None, f64::INFINITY);
    }
    let p0 = pts[0];
    let p1 = pts[pts.len() - 1];
    let chord = sub(p1, p0);
    let l = norm2(chord);
    if l < EPS_F {
        return (None, f64::INFINITY);
    }
    let mid = pts[pts.len() / 2];
    let side_mid = chord[0] * (mid[1] - p0[1]) - chord[1] * (mid[0] - p0[0]);
    let side_center = chord[0] * (c[1] - p0[1]) - chord[1] * (c[0] - p0[0]);
    let half_chord_over_r = (l / (2.0 * r)).clamp(-1.0, 1.0);
    let half_sweep_minor = half_chord_over_r.lasin();
    let is_major = side_mid * side_center > 0.0;
    let sweep_mag = if is_major { 2.0 * (std::f64::consts::PI - half_sweep_minor) } else { 2.0 * half_sweep_minor };
    if sweep_mag > 240.0f64.to_radians() {
        return (None, f64::INFINITY);
    }
    let sweep_sign = if side_mid > 0.0 { -1.0 } else { 1.0 };
    let sweep = sweep_sign * sweep_mag;
    let bulge = (sweep / 4.0).ltan();
    let sagitta = r * (1.0 - (0.5 * sweep_mag).lcos());
    if sagitta < MIN_ARC_SAGITTA_PX {
        return (None, f64::INFINITY);
    }
    (Some(Prim::Arc { p0, p1, bulge }), rms)
}

pub fn fit_full_circle(pts: &[P2]) -> (Option<Prim>, f64) {
    if pts.len() < 3 {
        return (None, f64::INFINITY);
    }
    let (c, r, rms) = fit_circle(pts);
    if !r.is_finite() || r < EPS_F {
        return (None, f64::INFINITY);
    }
    if r > MAX_RADIUS_FACTOR * segment_extent(pts) {
        return (None, f64::INFINITY);
    }
    (Some(Prim::Circle { center: c, radius: r }), rms)
}

#[derive(Clone, Debug)]
pub struct ChainPiece {
    pub start_idx: usize,
    pub end_idx: usize,
    pub primitive: Prim,
}

pub fn fit_single_primitive(pts: &[P2], line_tol_abs: f64, arc_tol_abs: f64) -> (Option<Prim>, f64) {
    let n = pts.len();
    if n < 2 {
        return (None, f64::INFINITY);
    }
    let (lp0, lp1, line_rms) = fit_line(pts);
    let line = Prim::Line { p0: lp0, p1: lp1 };
    let line_ok = line_rms < line_tol_abs;
    let line_sse = line_rms * line_rms * n as f64;
    if n < 3 {
        if line_ok {
            return (Some(line), line_sse);
        }
        return (None, f64::INFINITY);
    }
    let (arc, arc_rms) = fit_arc(pts);
    let arc_ok = arc.is_some() && arc_rms < arc_tol_abs;
    let arc_sse = if arc.is_some() { arc_rms * arc_rms * n as f64 } else { f64::INFINITY };
    let arc_min_sweep = 10.0f64.to_radians();
    if arc_ok {
        if let Some(Prim::Arc { bulge, .. }) = &arc {
            if arc_sweep(*bulge).abs() >= arc_min_sweep {
                return (arc, arc_sse);
            }
        }
    }
    if line_ok && (!arc_ok || line_sse <= 1.5 * arc_sse) {
        return (Some(line), line_sse);
    }
    if arc_ok {
        return (arc, arc_sse);
    }
    if line_ok {
        return (Some(line), line_sse);
    }
    (None, f64::INFINITY)
}

pub fn fit_segment_topdown(pts: &[P2], line_tol_abs: f64, arc_tol_abs: f64, min_len: usize) -> Vec<ChainPiece> {
    let max_depth = 12;
    let mut pieces: Vec<ChainPiece> = Vec::new();
    fn recurse(
        pts: &[P2],
        lo: usize,
        hi: usize,
        depth: usize,
        min_len: usize,
        max_depth: usize,
        line_tol_abs: f64,
        arc_tol_abs: f64,
        pieces: &mut Vec<ChainPiece>,
    ) {
        if hi - lo < min_len || depth >= max_depth {
            let (prim, _) = fit_single_primitive(&pts[lo..hi], line_tol_abs * 4.0, arc_tol_abs * 4.0);
            let prim = prim.unwrap_or(Prim::Line { p0: pts[lo], p1: pts[hi - 1] });
            pieces.push(ChainPiece { start_idx: lo, end_idx: hi, primitive: prim });
            return;
        }
        let (prim, _) = fit_single_primitive(&pts[lo..hi], line_tol_abs, arc_tol_abs);
        if let Some(p) = prim {
            pieces.push(ChainPiece { start_idx: lo, end_idx: hi, primitive: p });
            return;
        }
        let (lp0, lp1, _) = fit_line(&pts[lo..hi]);
        let kk = hi - lo;
        let line_res: Vec<f64> = pts[lo..hi].iter().map(|&p| line_perp_distance_k(lp0, lp1, p, kk).abs()).collect();
        let res: Vec<f64> = if hi - lo >= 3 {
            let (c, r, _) = fit_circle(&pts[lo..hi]);
            pts[lo..hi].iter().zip(&line_res).map(|(&p, &lr)| lr.min((crate::numeric::dist_naive(p, c) - r).abs())).collect()
        } else {
            line_res
        };
        let worst0 = crate::numeric::np_argmax(&res) + lo;
        // max(lo + min_len, min(hi - min_len, worst))
        let worst = (lo + min_len).max((hi - min_len).min(worst0));
        if worst <= lo || worst + 1 >= hi {
            let (prim, _) = fit_single_primitive(&pts[lo..hi], line_tol_abs * 4.0, arc_tol_abs * 4.0);
            let prim = prim.unwrap_or(Prim::Line { p0: lp0, p1: lp1 });
            pieces.push(ChainPiece { start_idx: lo, end_idx: hi, primitive: prim });
            return;
        }
        recurse(pts, lo, worst + 1, depth + 1, min_len, max_depth, line_tol_abs, arc_tol_abs, pieces);
        recurse(pts, worst, hi, depth + 1, min_len, max_depth, line_tol_abs, arc_tol_abs, pieces);
    }
    recurse(pts, 0, pts.len(), 0, min_len, max_depth, line_tol_abs, arc_tol_abs, &mut pieces);
    pieces.sort_by_key(|p| p.start_idx);
    pieces
}

fn validate_subloop_candidate(
    pts: &[P2],
    i: usize,
    j: usize,
    stride: usize,
    min_size: usize,
    max_circle_rms_rel: f64,
) -> Option<(usize, usize)> {
    let n = pts.len();
    let sub_ = &pts[i..=j];
    let (circle, rms) = fit_full_circle(sub_);
    circle.as_ref()?;
    let loop_ext = segment_extent(sub_);
    if rms >= max_circle_rms_rel * loop_ext {
        return None;
    }
    let xs: Vec<f64> = sub_.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = sub_.iter().map(|p| p[1]).collect();
    let bbox_x = np_max(&xs) - np_min(&xs);
    let bbox_y = np_max(&ys) - np_min(&ys);
    if bbox_x.min(bbox_y) < EPS_F {
        return None;
    }
    let aspect = bbox_x.max(bbox_y) / bbox_x.min(bbox_y);
    if aspect >= 1.25 {
        return None;
    }
    let mut best_d = dist(pts[j], pts[i]);
    let (mut ri, mut rj) = (i, j);
    let st = stride as i64;
    for di in -st..=st {
        let ii = i as i64 + di;
        if ii < 0 || ii >= n as i64 {
            continue;
        }
        for dj in -st..=st {
            let jj = j as i64 + dj;
            if jj < 0 || jj >= n as i64 || jj - ii < min_size as i64 {
                continue;
            }
            let d = dist(pts[jj as usize], pts[ii as usize]);
            if d < best_d {
                best_d = d;
                ri = ii as usize;
                rj = jj as usize;
            }
        }
    }
    Some((ri, rj))
}

pub fn find_closed_subloops(pts: &[P2]) -> Vec<(usize, usize)> {
    let min_size = 100usize;
    let closure_threshold_rel = 0.06;
    let min_extent_rel = 0.20;
    let max_circle_rms_rel = 0.06;
    let n = pts.len();
    if n < 2 * min_size {
        return vec![];
    }
    let extent = segment_extent(pts);
    if extent < 1.0 {
        return vec![];
    }
    let closure_thresh = closure_threshold_rel * extent;
    let extent_thresh = min_extent_rel * extent;
    let stride = 1usize.max(n / 200);
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < n - min_size {
        let mut j = n as i64 - 1;
        while j > (i + min_size) as i64 {
            let ju = j as usize;
            let d = dist(pts[ju], pts[i]);
            if d < closure_thresh {
                let sub_ = &pts[i..=ju];
                let sub_ext = segment_extent(sub_);
                if sub_ext >= extent_thresh {
                    candidates.push((ju - i, i, ju));
                }
                break;
            }
            j -= stride as i64;
        }
        i += stride;
    }
    if candidates.is_empty() {
        return vec![];
    }
    candidates.sort();
    candidates.reverse();
    let mut accepted: Vec<(usize, usize)> = Vec::new();
    for &(_size, i, j) in &candidates {
        if accepted.iter().any(|&(ai, aj)| !(j < ai || i > aj)) {
            continue;
        }
        let refined = match validate_subloop_candidate(pts, i, j, stride, min_size, max_circle_rms_rel) {
            Some(r) => r,
            None => continue,
        };
        let (ri, rj) = refined;
        if accepted.iter().any(|&(ai, aj)| !(rj < ai || ri > aj)) {
            continue;
        }
        accepted.push((ri, rj));
    }
    accepted.sort();
    accepted
}

fn line_collapse(pieces: Vec<ChainPiece>, src: Vec<Vec<P2>>, line_tol_abs: f64) -> (Vec<ChainPiece>, Vec<Vec<P2>>) {
    let n = pieces.len();
    if n < 2 {
        return (pieces, src);
    }
    let arc_sign_threshold = 8.0f64.to_radians();
    let sweep_signs: Vec<i32> = pieces
        .iter()
        .map(|c| match &c.primitive {
            Prim::Arc { bulge, .. } if arc_sweep(*bulge).abs() >= arc_sign_threshold => {
                if arc_sweep(*bulge) > 0.0 {
                    1
                } else {
                    -1
                }
            }
            _ => 0,
        })
        .collect();
    let mut out_pieces = Vec::new();
    let mut out_src = Vec::new();
    let mut i = 0;
    while i < n {
        if pieces[i].primitive.is_circle() {
            out_pieces.push(pieces[i].clone());
            out_src.push(src[i].clone());
            i += 1;
            continue;
        }
        let mut best_j = i + 1;
        let mut best_line: Option<(P2, P2)> = None;
        let mut best_pts: Option<Vec<P2>> = None;
        let mut seen_sign = sweep_signs[i];
        for j in i + 2..=n {
            if pieces[j - 1].primitive.is_circle() {
                break;
            }
            let nxt = sweep_signs[j - 1];
            if nxt != 0 {
                if seen_sign != 0 && nxt != seen_sign {
                    break;
                }
                seen_sign = nxt;
            }
            let concat: Vec<P2> = src[i..j].iter().flatten().cloned().collect();
            if concat.len() < 2 {
                break;
            }
            let (lp0, lp1, rms) = fit_line(&concat);
            if rms >= line_tol_abs {
                break;
            }
            best_j = j;
            best_line = Some((lp0, lp1));
            best_pts = Some(concat);
        }
        if let (Some((p0, p1)), Some(bp)) = (best_line, best_pts) {
            out_pieces.push(ChainPiece {
                start_idx: pieces[i].start_idx,
                end_idx: pieces[best_j - 1].end_idx,
                primitive: Prim::Line { p0, p1 },
            });
            out_src.push(bp);
        } else {
            out_pieces.push(pieces[i].clone());
            out_src.push(src[i].clone());
        }
        i = best_j;
    }
    (out_pieces, out_src)
}

fn is_degenerate(p: &Prim) -> bool {
    match p {
        Prim::Line { p0, p1 } => line_length(*p0, *p1) < 0.5,
        Prim::Arc { bulge, .. } => arc_sweep(*bulge).abs() < 0.5f64.to_radians(),
        _ => false,
    }
}

fn drop_degenerate_pieces(
    pieces: &[ChainPiece],
    src: &[Vec<P2>],
    prims: &[Prim],
) -> (Vec<ChainPiece>, Vec<Vec<P2>>, Vec<Prim>) {
    let mut out_pieces: Vec<ChainPiece> = Vec::new();
    let mut out_src: Vec<Vec<P2>> = Vec::new();
    let mut out_prims: Vec<Prim> = Vec::new();
    let mut pending_src: Vec<P2> = Vec::new();
    let mut pending_any = false;
    let mut pending_start: usize = 0;
    for (k, prim) in prims.iter().enumerate() {
        if is_degenerate(prim) {
            if !out_pieces.is_empty() {
                let last = out_pieces.last_mut().unwrap();
                last.end_idx = pieces[k].end_idx;
                out_src.last_mut().unwrap().extend_from_slice(&src[k]);
            } else {
                if !pending_any {
                    pending_start = pieces[k].start_idx;
                    pending_any = true;
                }
                pending_src.extend_from_slice(&src[k]);
            }
            continue;
        }
        if pending_any {
            let mut s = pending_src.clone();
            s.extend_from_slice(&src[k]);
            out_pieces.push(ChainPiece { start_idx: pending_start, end_idx: pieces[k].end_idx, primitive: prim.clone() });
            out_src.push(s);
            pending_src.clear();
            pending_any = false;
        } else {
            out_pieces.push(pieces[k].clone());
            out_src.push(src[k].clone());
        }
        out_prims.push(prim.clone());
    }
    (out_pieces, out_src, out_prims)
}

pub fn fuse_chain(pieces: &[ChainPiece], src: &[Vec<P2>], prims: &[Prim]) -> (Vec<ChainPiece>, Vec<Vec<P2>>) {
    let (line_tol_rel, arc_tol_rel, line_tol_abs_min, arc_tol_abs_min) = (0.006, 0.025, 1.0, 2.5);
    let (pieces, src, prims) = drop_degenerate_pieces(pieces, src, prims);
    let n = pieces.len();
    if n <= 1 {
        if n == 0 {
            return (vec![], vec![]);
        }
        return (
            vec![ChainPiece { start_idx: pieces[0].start_idx, end_idx: pieces[0].end_idx, primitive: prims[0].clone() }],
            vec![src[0].clone()],
        );
    }
    let all_pts: Vec<P2> = src.iter().flatten().cloned().collect();
    let extent = segment_extent(&all_pts);
    let line_tol_abs = (line_tol_rel * extent).max(line_tol_abs_min);
    let arc_tol_abs = (arc_tol_rel * extent).max(arc_tol_abs_min);
    let mut new_pieces: Vec<ChainPiece> = Vec::new();
    let mut new_src: Vec<Vec<P2>> = Vec::new();
    let mut i = 0;
    while i < n {
        if prims[i].is_circle() {
            new_pieces.push(ChainPiece { start_idx: pieces[i].start_idx, end_idx: pieces[i].end_idx, primitive: prims[i].clone() });
            new_src.push(src[i].clone());
            i += 1;
            continue;
        }
        let mut best_j = i;
        let mut best_prim: Option<Prim> = None;
        let mut best_pts: Option<Vec<P2>> = None;
        for j in i + 1..=n {
            if j > i + 1 && prims[j - 1].is_circle() {
                break;
            }
            let concat: Vec<P2> = src[i..j].iter().flatten().cloned().collect();
            if concat.len() < 2 {
                break;
            }
            if j > i + 1 && has_curvature_reversal(&concat) {
                break;
            }
            let (lp0, lp1, line_rms) = fit_line(&concat);
            let (arc, arc_rms) = if concat.len() >= 3 { fit_arc(&concat) } else { (None, f64::INFINITY) };
            let arc_ok = arc.is_some() && arc_rms < arc_tol_abs;
            let arc_meaningful = arc_ok
                && match &arc {
                    Some(Prim::Arc { bulge, .. }) => arc_sweep(*bulge).abs() >= 10.0f64.to_radians(),
                    _ => false,
                };
            let candidate = if arc_meaningful {
                arc
            } else if line_rms < line_tol_abs {
                Some(Prim::Line { p0: lp0, p1: lp1 })
            } else if arc_ok {
                arc
            } else {
                None
            };
            match candidate {
                None => break,
                Some(c) => {
                    best_j = j;
                    best_prim = Some(c);
                    best_pts = Some(concat);
                }
            }
        }
        if let (Some(bp), Some(bpts)) = (best_prim, best_pts) {
            new_pieces.push(ChainPiece { start_idx: pieces[i].start_idx, end_idx: pieces[best_j - 1].end_idx, primitive: bp });
            new_src.push(bpts);
            i = best_j;
        } else {
            new_pieces.push(ChainPiece { start_idx: pieces[i].start_idx, end_idx: pieces[i].end_idx, primitive: prims[i].clone() });
            new_src.push(src[i].clone());
            i += 1;
        }
    }
    line_collapse(new_pieces, new_src, line_tol_abs)
}

pub struct FitParams {
    pub line_tol: f64,
    pub arc_tol: f64,
    pub min_len: usize,
    pub circle_rms_rel: f64,
}

pub fn fit_polyline(pts: &[P2], fp: &FitParams) -> Vec<ChainPiece> {
    let n = pts.len();
    if n < 2 {
        return vec![];
    }
    let extent = segment_extent(pts);
    let line_tol_abs = fp.line_tol * extent;
    let arc_tol_abs = fp.arc_tol * extent;
    let circle_rms_abs = fp.circle_rms_rel * extent;
    let corners = find_corners(pts);
    let near_closed = is_near_closed_polyline(pts, 0.10, 10.0);
    if near_closed && corners.is_empty() {
        let (circle, rms) = fit_full_circle(pts);
        if let Some(c) = circle {
            if rms < circle_rms_abs {
                let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
                let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
                let bbox_x = np_max(&xs) - np_min(&xs);
                let bbox_y = np_max(&ys) - np_min(&ys);
                let aspect = if bbox_x.min(bbox_y) > EPS_F { bbox_x.max(bbox_y) / bbox_x.min(bbox_y) } else { f64::INFINITY };
                if aspect < 1.25 {
                    return vec![ChainPiece { start_idx: 0, end_idx: n, primitive: c }];
                }
            }
        }
    }
    let subloops = if !near_closed { find_closed_subloops(pts) } else { vec![] };
    let mut splits: Vec<usize> = corners.clone();
    for (si, sj) in subloops {
        splits.push(si);
        splits.push(sj);
    }
    let inflections = find_inflections(pts);
    splits.extend(inflections);
    splits.sort();
    splits.dedup();
    if !splits.is_empty() {
        let mut pieces = Vec::new();
        let mut split_idxs = vec![0usize];
        split_idxs.extend(splits.iter().cloned());
        split_idxs.push(n);
        split_idxs.sort();
        split_idxs.dedup();
        for k in 0..split_idxs.len() - 1 {
            let lo = split_idxs[k];
            let hi = split_idxs[k + 1];
            let sub_ = if hi < n { &pts[lo..hi + 1] } else { &pts[lo..] };
            if sub_.len() < 2 {
                continue;
            }
            for cp in fit_polyline(sub_, fp) {
                pieces.push(ChainPiece { start_idx: cp.start_idx + lo, end_idx: cp.end_idx + lo, primitive: cp.primitive });
            }
        }
        return pieces;
    }
    let arc_rms_strict = (0.04 * extent).min(SINGLE_ARC_SHORTCUT_RMS_CAP);
    let arc_rms_loose = (0.04 * extent).min(SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP);
    if corners.is_empty() {
        let (arc, arc_rms) = fit_arc(pts);
        if let Some(Prim::Arc { p0, p1, bulge }) = &arc {
            if arc_chord(*p0, *p1) > line_tol_abs * 2.0 {
                if arc_rms < arc_rms_strict {
                    return vec![ChainPiece { start_idx: 0, end_idx: n, primitive: Prim::Arc { p0: *p0, p1: *p1, bulge: *bulge } }];
                }
                if arc_rms < arc_rms_loose && n < SUBDIVISION_MIN_POINTS {
                    return vec![ChainPiece { start_idx: 0, end_idx: n, primitive: Prim::Arc { p0: *p0, p1: *p1, bulge: *bulge } }];
                }
            }
        }
    }
    let (single, _) = fit_single_primitive(pts, line_tol_abs, arc_tol_abs);
    if let Some(s) = single {
        return vec![ChainPiece { start_idx: 0, end_idx: n, primitive: s }];
    }
    fit_segment_topdown(pts, line_tol_abs, arc_tol_abs, fp.min_len)
}
