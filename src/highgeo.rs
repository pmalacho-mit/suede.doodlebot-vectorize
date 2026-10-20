//! High-geometry (naive baseline) vectorizer (`vectorize/high_geometry/__init__.py`).

use crate::fmath::DetMath;
use crate::commands::Command;
use crate::config::Config;
use crate::ndimage::{find_peaks, gaussian_filter1d};
use crate::numeric::{
    dist, mean_axis0, norm2, np_gradient, np_interp, np_linspace, np_max, np_searchsorted_left, np_sum,
    np_unwrap, py_mod, solve2, sub, svd_vt_n2, P2,
};
use crate::segment::Polyline;

#[derive(Clone, Debug)]
enum HPrim {
    Line { start: P2, end: P2 },
    Arc { center: P2, radius: f64, start: P2, end: P2, ccw: bool },
}
impl HPrim {
    fn start(&self) -> P2 {
        match self {
            HPrim::Line { start, .. } | HPrim::Arc { start, .. } => *start,
        }
    }
    fn end(&self) -> P2 {
        match self {
            HPrim::Line { end, .. } | HPrim::Arc { end, .. } => *end,
        }
    }
}

fn resample_polyline(poly: &[P2], n: usize) -> (Vec<P2>, Vec<f64>) {
    let seg: Vec<f64> = (1..poly.len()).map(|i| crate::numeric::dist_naive(poly[i], poly[i - 1])).collect();
    let mut s = vec![0.0];
    let mut acc = 0.0;
    for v in &seg {
        acc += v;
        s.push(acc);
    }
    let total = *s.last().unwrap();
    if total < 1e-9 {
        return (poly.to_vec(), s);
    }
    let s_new = np_linspace(0.0, total, n);
    let xs: Vec<f64> = poly.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = poly.iter().map(|p| p[1]).collect();
    let pts: Vec<P2> = s_new.iter().map(|&t| [np_interp(t, &s, &xs), np_interp(t, &s, &ys)]).collect();
    (pts, s_new)
}

fn curvature(xy: &[P2], sigma: f64) -> Vec<f64> {
    let xs: Vec<f64> = xy.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = xy.iter().map(|p| p[1]).collect();
    let xs = gaussian_filter1d(&xs, sigma);
    let ys = gaussian_filter1d(&ys, sigma);
    let dx = np_gradient(&xs);
    let dy = np_gradient(&ys);
    let ddx = np_gradient(&dx);
    let ddy = np_gradient(&dy);
    (0..xy.len())
        .map(|i| {
            let mut denom = (dx[i] * dx[i] + dy[i] * dy[i]).lpowf(1.5);
            if denom < 1e-10 {
                denom = 1e-10;
            }
            (dx[i] * ddy[i] - dy[i] * ddx[i]) / denom
        })
        .collect()
}

fn segment_at_corners(poly: &[P2], sigma: f64, corner_threshold: f64) -> Vec<Vec<P2>> {
    let min_segment_length = 4usize;
    if poly.len() < 2 * min_segment_length {
        return vec![poly.to_vec()];
    }
    let seg: Vec<f64> = (1..poly.len()).map(|i| crate::numeric::dist_naive(poly[i], poly[i - 1])).collect();
    let total_len = np_sum(&seg);
    if total_len < 4.0 {
        return vec![poly.to_vec()];
    }
    let n_samples = (total_len as i64 as usize).max(16);
    let (resampled, s_res) = resample_polyline(poly, n_samples);
    let kappa = curvature(&resampled, sigma);
    let abs_k: Vec<f64> = kappa.iter().map(|v| v.abs()).collect();
    let peaks = find_peaks(&abs_k, corner_threshold, ((sigma * 3.0) as i64).max(3) as f64);
    if peaks.is_empty() {
        return vec![poly.to_vec()];
    }
    let mut s_orig = vec![0.0];
    let mut acc = 0.0;
    for v in &seg {
        acc += v;
        s_orig.push(acc);
    }
    let mut cut_points: Vec<usize> = Vec::new();
    for p in peaks {
        let s_at = s_res[p];
        let idx = np_searchsorted_left(&s_orig, s_at) as i64;
        let idx = idx.max(1).min(poly.len() as i64 - 2) as usize;
        cut_points.push(idx);
    }
    cut_points.sort();
    cut_points.dedup();
    let mut segments: Vec<Vec<P2>> = Vec::new();
    let mut prev = 0usize;
    for ci in cut_points {
        if ci - prev >= min_segment_length {
            segments.push(poly[prev..=ci].to_vec());
            prev = ci;
        }
    }
    if poly.len() - prev >= min_segment_length {
        segments.push(poly[prev..].to_vec());
    }
    if segments.is_empty() {
        vec![poly.to_vec()]
    } else {
        segments
    }
}

fn fit_line_h(pts: &[P2]) -> f64 {
    let centroid = mean_axis0(pts);
    let centered: Vec<P2> = pts.iter().map(|&p| sub(p, centroid)).collect();
    let vt = svd_vt_n2(&centered);
    let perp = vt[1];
    let k = centered.len();
    let d: Vec<f64> = centered.iter().map(|&c| crate::blas::gemv_row(c, perp, k).abs()).collect();
    np_max(&d)
}

fn fit_circle_algebraic(pts: &[P2]) -> Option<(f64, f64, f64)> {
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
    let x_m = crate::numeric::np_mean(&xs);
    let y_m = crate::numeric::np_mean(&ys);
    let u: Vec<f64> = xs.iter().map(|x| x - x_m).collect();
    let v: Vec<f64> = ys.iter().map(|y| y - y_m).collect();
    let prod = |f: &dyn Fn(usize) -> f64| -> f64 {
        let a: Vec<f64> = (0..pts.len()).map(f).collect();
        np_sum(&a)
    };
    let suu = prod(&|i| u[i] * u[i]);
    let svv = prod(&|i| v[i] * v[i]);
    let suv = prod(&|i| u[i] * v[i]);
    let suuu = prod(&|i| u[i] * u[i] * u[i]);
    let svvv = prod(&|i| v[i] * v[i] * v[i]);
    let suvv = prod(&|i| u[i] * v[i] * v[i]);
    let svuu = prod(&|i| v[i] * u[i] * u[i]);
    let a = [[suu, suv], [suv, svv]];
    let b = [0.5 * (suuu + suvv), 0.5 * (svvv + svuu)];
    let sol = solve2(a, b)?;
    let (uc, vc) = (sol[0], sol[1]);
    let cx = uc + x_m;
    let cy = vc + y_m;
    let r = (uc * uc + vc * vc + (suu + svv) / pts.len() as f64).sqrt();
    Some((cx, cy, r))
}

fn arc_traversal_ccw(pts: &[P2], cx: f64, cy: f64) -> bool {
    let angles: Vec<f64> = pts.iter().map(|p| (p[1] - cy).latan2(p[0] - cx)).collect();
    let un = np_unwrap(&angles);
    un[un.len() - 1] - un[0] > 0.0
}

struct ArcFit {
    center: P2,
    radius: f64,
    residual: f64,
    ccw: bool,
}

fn fit_arc_h(pts: &[P2]) -> Option<ArcFit> {
    if pts.len() < 3 {
        return None;
    }
    let (cx, cy, r) = fit_circle_algebraic(pts)?;
    if !r.is_finite() || r < 0.5 {
        return None;
    }
    let res: Vec<f64> = pts
        .iter()
        .map(|p| (((p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy)).sqrt() - r).abs())
        .collect();
    Some(ArcFit { center: [cx, cy], radius: r, residual: np_max(&res), ccw: arc_traversal_ccw(pts, cx, cy) })
}

fn segment_scale(pts: &[P2]) -> f64 {
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
    let bbox = [np_max(&xs) - crate::numeric::np_min(&xs), np_max(&ys) - crate::numeric::np_min(&ys)];
    norm2(bbox).max(1.0)
}

fn project_to_circle(p: P2, center: P2, r: f64) -> P2 {
    let d = sub(p, center);
    let n = norm2(d);
    if n < 1e-9 {
        return p;
    }
    [center[0] + (r / n) * d[0], center[1] + (r / n) * d[1]]
}

fn best_fit_with_residual(pts: &[P2]) -> (HPrim, f64) {
    let (arc_advantage, max_radius_factor, min_arc_radius) = (1.5, 8.0, 1.5);
    let line_res = fit_line_h(pts);
    let scale = segment_scale(pts);
    let arc = fit_arc_h(pts);
    if let Some(a) = arc {
        let use_arc = a.radius >= min_arc_radius && a.radius <= max_radius_factor * scale && a.residual * arc_advantage < line_res.max(1e-6);
        if use_arc {
            return (
                HPrim::Arc {
                    center: a.center,
                    radius: a.radius,
                    start: project_to_circle(pts[0], a.center, a.radius),
                    end: project_to_circle(pts[pts.len() - 1], a.center, a.radius),
                    ccw: a.ccw,
                },
                a.residual,
            );
        }
    }
    (HPrim::Line { start: pts[0], end: pts[pts.len() - 1] }, line_res)
}

fn polyline_to_stroke(poly: &[P2], sigma: f64, corner_threshold: f64, max_fit_residual: f64) -> Vec<HPrim> {
    let min_split_length = 4usize;
    let segments = segment_at_corners(poly, sigma, corner_threshold);
    let mut stack: Vec<Vec<P2>> = segments.into_iter().filter(|s| s.len() >= 2).collect();
    stack.reverse();
    let mut prims = Vec::new();
    while let Some(seg) = stack.pop() {
        let (prim, residual) = best_fit_with_residual(&seg);
        if residual <= max_fit_residual || seg.len() < 2 * min_split_length {
            prims.push(prim);
            continue;
        }
        let mid = seg.len() / 2;
        let left = seg[..mid + 1].to_vec();
        let right = seg[mid..].to_vec();
        stack.push(right);
        stack.push(left);
    }
    prims
}

fn reverse_prim(p: &HPrim) -> HPrim {
    match p {
        HPrim::Line { start, end } => HPrim::Line { start: *end, end: *start },
        HPrim::Arc { center, radius, start, end, ccw } => {
            HPrim::Arc { center: *center, radius: *radius, start: *end, end: *start, ccw: !ccw }
        }
    }
}

fn order_strokes(strokes: Vec<Vec<HPrim>>, origin: P2) -> Vec<Vec<HPrim>> {
    let mut remaining: Vec<usize> = (0..strokes.len()).collect();
    let mut cursor = origin;
    let mut ordered = Vec::new();
    while !remaining.is_empty() {
        let mut best_i = remaining[0];
        let mut best_dist = f64::INFINITY;
        let mut best_reverse = false;
        for &i in &remaining {
            let s = strokes[i][0].start();
            let e = strokes[i][strokes[i].len() - 1].end();
            let ds = dist(cursor, s);
            let de = dist(cursor, e);
            if ds < best_dist {
                best_dist = ds;
                best_i = i;
                best_reverse = false;
            }
            if de < best_dist {
                best_dist = de;
                best_i = i;
                best_reverse = true;
            }
        }
        let chosen: Vec<HPrim> = if best_reverse {
            strokes[best_i].iter().rev().map(reverse_prim).collect()
        } else {
            strokes[best_i].clone()
        };
        cursor = chosen[chosen.len() - 1].end();
        ordered.push(chosen);
        remaining.retain(|&x| x != best_i);
    }
    ordered
}

fn wrap_pi(a: f64) -> f64 {
    py_mod(a + std::f64::consts::PI, 2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

fn emit_spin_to(target: f64, heading: f64) -> (Option<Command>, f64) {
    let delta = wrap_pi(target - heading);
    if delta.abs() < 1e-4 {
        return (None, heading);
    }
    (Some(Command::Spin { degrees: delta.to_degrees() }), target)
}

fn emit_primitive(prim: &HPrim, pos: P2, heading: f64, out: &mut Vec<Command>) -> (P2, f64) {
    match prim {
        HPrim::Line { end, .. } => {
            let delta = sub(*end, pos);
            let d = norm2(delta);
            if d < 1e-6 {
                return (pos, heading);
            }
            let target = delta[1].latan2(delta[0]);
            let (spin, heading) = emit_spin_to(target, heading);
            if let Some(s) = spin {
                out.push(s);
            }
            out.push(Command::Line { distance: d, pen_down: true });
            (*end, heading)
        }
        HPrim::Arc { center, radius, start, end, ccw } => {
            let radial = sub(*start, *center);
            let tangent = if *ccw { [-radial[1], radial[0]] } else { [radial[1], -radial[0]] };
            let target = tangent[1].latan2(tangent[0]);
            let (spin, heading) = emit_spin_to(target, heading);
            if let Some(s) = spin {
                out.push(s);
            }
            let start_a = (start[1] - center[1]).latan2(start[0] - center[0]);
            let end_a = (end[1] - center[1]).latan2(end[0] - center[0]);
            let mut sweep = end_a - start_a;
            if *ccw {
                if sweep <= 0.0 {
                    sweep += 2.0 * std::f64::consts::PI;
                }
            } else if sweep >= 0.0 {
                sweep -= 2.0 * std::f64::consts::PI;
            }
            out.push(Command::Arc { radius: *radius, degrees: sweep.to_degrees() });
            (*end, wrap_pi(heading + sweep))
        }
    }
}

pub fn vectorize_high_geometry(polylines: &[Polyline], start_pos: P2, start_heading: f64, cfg: &Config) -> Vec<Command> {
    let strokes: Vec<Vec<HPrim>> = polylines
        .iter()
        .map(|p| polyline_to_stroke(p, cfg.hg_sigma, cfg.hg_corner_threshold, cfg.hg_max_fit_residual))
        .filter(|s| !s.is_empty())
        .collect();
    let ordered = order_strokes(strokes, start_pos);
    let mut pos = start_pos;
    let mut heading = start_heading;
    let mut out = Vec::new();
    for stroke in ordered {
        let target = stroke[0].start();
        let delta = sub(target, pos);
        let d = norm2(delta);
        if d > 1e-6 {
            let th = delta[1].latan2(delta[0]);
            let (spin, h) = emit_spin_to(th, heading);
            heading = h;
            if let Some(s) = spin {
                out.push(s);
            }
            out.push(Command::Line { distance: d, pen_down: false });
            pos = target;
        }
        for prim in &stroke {
            let (p, h) = emit_primitive(prim, pos, heading, &mut out);
            pos = p;
            heading = h;
        }
    }
    out
}
