//! Beautification detection and arc-pair merging (`beautify.py`).

use crate::fmath::DetMath;
use super::primitives::*;
use super::solve::SoftConstraints;
use crate::numeric::{dist, mean_axis0, np_mean, py_mod, P2};

pub struct BeautifyTolerances {
    pub parallel_rad: f64,
    pub perp_rad: f64,
    pub radius_rel: f64,
    pub center_abs: f64,
    pub min_radius: f64,
    pub min_line_length: f64,
}
impl Default for BeautifyTolerances {
    fn default() -> Self {
        BeautifyTolerances { parallel_rad: 0.10, perp_rad: 0.10, radius_rel: 0.07, center_abs: 4.0, min_radius: 2.0, min_line_length: 5.0 }
    }
}

fn line_angle(p0: P2, p1: P2) -> f64 {
    let d = line_direction(p0, p1);
    d[1].latan2(d[0])
}

pub fn detect(prims: &[Prim], tol: &BeautifyTolerances) -> SoftConstraints {
    let mut out = SoftConstraints::default();
    let pi = std::f64::consts::PI;
    let lines: Vec<(usize, P2, P2)> = prims
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Prim::Line { p0, p1 } if line_length(*p0, *p1) >= tol.min_line_length => Some((i, *p0, *p1)),
            _ => None,
        })
        .collect();
    for a in 0..lines.len() {
        for b in a + 1..lines.len() {
            let (i, a0, a1) = lines[a];
            let (j, b0, b1) = lines[b];
            let ai = line_angle(a0, a1);
            let aj = line_angle(b0, b1);
            let delta = py_mod(ai - aj + pi / 2.0, pi) - pi / 2.0;
            let absdelta = delta.abs();
            if absdelta < tol.parallel_rad {
                out.parallel.push((i, j));
            } else if (absdelta - pi / 2.0).abs() < tol.perp_rad {
                out.perpendicular.push((i, j));
            }
        }
    }
    let curved: Vec<usize> = prims
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.is_arc() || p.is_circle() {
                let r = prim_radius(p);
                if r >= tol.min_radius && r.is_finite() {
                    return Some(i);
                }
            }
            None
        })
        .collect();
    for a in 0..curved.len() {
        for b in a + 1..curved.len() {
            let i = curved[a];
            let j = curved[b];
            let ra = prim_radius(&prims[i]);
            let rb = prim_radius(&prims[j]);
            let denom = ra.max(rb);
            if denom > 0.0 && (ra - rb).abs() / denom < tol.radius_rel {
                out.equal_radius.push((i, j));
            }
            if let (Some(ca), Some(cb)) = (prim_center(&prims[i]), prim_center(&prims[j])) {
                if dist(ca, cb) < tol.center_abs {
                    out.concentric.push((i, j));
                }
            }
        }
    }
    out
}

fn arc_angular_coverage(p0: P2, _p1: P2, bulge: f64, center: P2) -> std::collections::HashSet<i64> {
    let a0 = (p0[1] - center[1]).latan2(p0[0] - center[0]).to_degrees();
    let sweep = arc_sweep(bulge).to_degrees();
    let lo = a0.min(a0 + sweep);
    let hi = a0.max(a0 + sweep);
    let mut bins = std::collections::HashSet::new();
    let mut d = lo.floor();
    while d <= hi {
        bins.insert(py_mod(d, 360.0) as i64);
        d += 1.0;
    }
    bins
}

pub fn merge_arc_pairs(prims: &[Prim]) -> Vec<Prim> {
    let center_tol_rel = 1.0;
    let radius_tol_rel = 0.10;
    let endpoint_tol_rel = 0.20;
    let full_circle_sweep_thresh_deg = 320.0;
    let sweep_match_tol_deg = 40.0;
    let n = prims.len();
    if n < 2 {
        return prims.to_vec();
    }
    let arc_pids: Vec<usize> = (0..n)
        .filter(|&i| prims[i].is_arc() && prim_radius(&prims[i]).is_finite() && prim_radius(&prims[i]) > 2.0)
        .collect();
    let mut used = vec![false; n];
    let mut merged_any = false;
    let mut replacement: Vec<Option<Option<Prim>>> = vec![None; n]; // Some(Some(p)) replace, Some(None) drop
    let arc_parts = |p: &Prim| -> (P2, P2, f64) {
        match p {
            Prim::Arc { p0, p1, bulge } => (*p0, *p1, *bulge),
            _ => unreachable!(),
        }
    };
    for ii in 0..arc_pids.len() {
        let i = arc_pids[ii];
        if used[i] {
            continue;
        }
        let (ap0, ap1, ab) = arc_parts(&prims[i]);
        let ra = prim_radius(&prims[i]);
        let ca = prim_center(&prims[i]).unwrap();
        for jj in ii + 1..arc_pids.len() {
            let j = arc_pids[jj];
            if used[j] {
                continue;
            }
            let (bp0, bp1, bb) = arc_parts(&prims[j]);
            let rb = prim_radius(&prims[j]);
            let cb = prim_center(&prims[j]).unwrap();
            let r_mean = 0.5 * (ra + rb);
            if (ra - rb).abs() / ra.max(rb) > radius_tol_rel {
                continue;
            }
            if dist(ca, cb) > center_tol_rel * r_mean {
                continue;
            }
            let tol_pt = (endpoint_tol_rel * r_mean).max(3.0);
            let d_e_s = dist(ap1, bp0);
            let d_e_e = dist(ap1, bp1);
            let d_s_s = dist(ap0, bp0);
            let d_s_e = dist(ap0, bp1);
            let chained = (d_e_s < tol_pt && d_s_e < tol_pt) || (d_e_e < tol_pt && d_s_s < tol_pt);
            let parallel = (d_s_s < tol_pt && d_e_e < tol_pt) || (d_s_e < tol_pt && d_e_s < tol_pt);
            if !(chained || parallel) {
                continue;
            }
            let sweep_a = arc_sweep(ab).to_degrees().abs();
            let sweep_b = arc_sweep(bb).to_degrees().abs();
            if sweep_a + sweep_b < full_circle_sweep_thresh_deg {
                continue;
            }
            if (sweep_a - sweep_b).abs() > sweep_match_tol_deg {
                continue;
            }
            let new_center = [0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1])];
            replacement[i] = Some(Some(Prim::Circle { center: new_center, radius: r_mean }));
            replacement[j] = Some(None);
            used[i] = true;
            used[j] = true;
            merged_any = true;
            break;
        }
    }
    // 3+ arc grouping
    let remaining: Vec<usize> = arc_pids.iter().cloned().filter(|&p| !used[p]).collect();
    for ii in 0..remaining.len() {
        let i = remaining[ii];
        if used[i] {
            continue;
        }
        let ra = prim_radius(&prims[i]);
        let ca = prim_center(&prims[i]).unwrap();
        let mut group = vec![i];
        for &j in &remaining[ii + 1..] {
            if used[j] {
                continue;
            }
            let rb = prim_radius(&prims[j]);
            let cb = prim_center(&prims[j]).unwrap();
            let r_mean = 0.5 * (ra + rb);
            if (ra - rb).abs() / ra.max(rb) > radius_tol_rel {
                continue;
            }
            if dist(ca, cb) > center_tol_rel * r_mean {
                continue;
            }
            group.push(j);
        }
        if group.len() < 3 {
            continue;
        }
        let radii: Vec<f64> = group.iter().map(|&g| prim_radius(&prims[g])).collect();
        let rmax = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rmin = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        if (rmax - rmin) / rmax > radius_tol_rel {
            continue;
        }
        let centers: Vec<P2> = group.iter().map(|&g| prim_center(&prims[g]).unwrap()).collect();
        let mean_c = mean_axis0(&centers);
        let mean_r = np_mean(&radii);
        let mut covered = std::collections::HashSet::new();
        let mut total_sweep = 0.0;
        for &g in &group {
            let (p0, p1, b) = arc_parts(&prims[g]);
            covered.extend(arc_angular_coverage(p0, p1, b, mean_c));
            total_sweep += arc_sweep(b).to_degrees().abs();
        }
        if covered.len() < 340 {
            continue;
        }
        if total_sweep > 420.0 {
            continue;
        }
        let tol_pt = (endpoint_tol_rel * mean_r).max(3.0);
        let mut endpoints: Vec<P2> = Vec::new();
        for &g in &group {
            let (p0, p1, _) = arc_parts(&prims[g]);
            endpoints.push(p0);
            endpoints.push(p1);
        }
        let mut connected = true;
        for (ei, e) in endpoints.iter().enumerate() {
            let partner = endpoints.iter().enumerate().any(|(eo, o)| (eo / 2) != (ei / 2) && dist(*e, *o) < tol_pt);
            if !partner {
                connected = false;
                break;
            }
        }
        if !connected {
            continue;
        }
        replacement[group[0]] = Some(Some(Prim::Circle { center: mean_c, radius: mean_r }));
        for &g in &group[1..] {
            replacement[g] = Some(None);
        }
        for &g in &group {
            used[g] = true;
        }
        merged_any = true;
    }
    if !merged_any {
        return prims.to_vec();
    }
    let mut out = Vec::new();
    for (pid, p) in prims.iter().enumerate() {
        match &replacement[pid] {
            Some(Some(rep)) => out.push(rep.clone()),
            Some(None) => {}
            None => out.push(p.clone()),
        }
    }
    out
}
