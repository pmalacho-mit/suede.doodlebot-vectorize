//! Stroke graph construction (`graph/__init__.py`).

use crate::fmath::DetMath;
use crate::config::Config;
use crate::numeric::{dist, dot, mean_axis0, norm2, py_floordiv, py_round, sub, svd_vt_n2, P2};
use crate::segment::Polyline;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Terminal,
    Crossing,
    Cusp,
}

#[derive(Clone, Debug)]
pub struct Participation {
    pub polyline_index: usize,
    pub point_index: usize,
    pub role: Role,
    pub terminal_end: Option<usize>,
    pub tangent_in: P2,
    pub tangent_out: Option<P2>,
}

#[derive(Clone, Debug)]
pub struct Junction {
    pub location: P2,
    pub participants: Vec<Participation>,
}

pub struct StrokeGraph {
    pub polylines: Vec<Polyline>,
    pub junctions: Vec<Junction>,
}

const CLOSED_TOL: f64 = 1.5;

pub fn is_closed_polyline(poly: &[P2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    dist(poly[0], poly[poly.len() - 1]) < CLOSED_TOL
}

fn pca_unit_tangent(samples: &[P2], orient_toward: P2) -> P2 {
    if samples.len() < 2 {
        return [1.0, 0.0];
    }
    let c = mean_axis0(samples);
    let centered: Vec<P2> = samples.iter().map(|&p| sub(p, c)).collect();
    let vt = svd_vt_n2(&centered);
    let mut t = vt[0];
    let n = norm2(t);
    if n < 1e-9 {
        return [1.0, 0.0];
    }
    t = [t[0] / n, t[1] / n];
    if dot(t, orient_toward) < 0.0 {
        t = [-t[0], -t[1]];
    }
    t
}

fn terminal_tangent(poly: &[P2], terminal_end: usize, window: usize) -> P2 {
    let n = poly.len();
    if n < 2 {
        return [1.0, 0.0];
    }
    if terminal_end == 0 {
        let samples = &poly[..n.min(window + 1)];
        let orient = sub(samples[samples.len() - 1], samples[0]);
        pca_unit_tangent(samples, orient)
    } else {
        let start = if n > window + 1 { n - window - 1 } else { 0 };
        let samples = &poly[start..];
        let orient = sub(samples[0], samples[samples.len() - 1]);
        pca_unit_tangent(samples, orient)
    }
}

fn is_integer_point(p: P2) -> bool {
    let tol = 0.05;
    (p[0] - py_round(p[0])).abs() < tol && (p[1] - py_round(p[1])).abs() < tol
}

fn crossing_tangents(poly: &[P2], idx: usize, skip: i64, half_window: i64) -> (P2, P2) {
    let n = poly.len() as i64;
    if n < 2 {
        return ([1.0, 0.0], [1.0, 0.0]);
    }
    let here = poly[idx];
    let find_anchor = |direction: i64| -> i64 {
        let mut cur = idx as i64 + direction * skip;
        cur = cur.max(0).min(n - 1);
        while 0 < cur && cur < n - 1 && !is_integer_point(poly[cur as usize]) {
            let nxt = cur + direction;
            if nxt < 0 || nxt >= n {
                break;
            }
            cur = nxt;
        }
        cur
    };
    let in_hi = find_anchor(-1);
    let in_lo = (in_hi - half_window).max(0);
    let tangent_in = if in_hi > in_lo {
        let samples = &poly[in_lo as usize..=in_hi as usize];
        pca_unit_tangent(samples, sub(poly[in_lo as usize], here))
    } else {
        [1.0, 0.0]
    };
    let out_lo = find_anchor(1);
    let out_hi = (out_lo + half_window).min(n - 1);
    let tangent_out = if out_hi > out_lo {
        let samples = &poly[out_lo as usize..=out_hi as usize];
        pca_unit_tangent(samples, sub(poly[out_hi as usize], here))
    } else {
        [1.0, 0.0]
    };
    (tangent_in, tangent_out)
}

fn closed_loop_tangents(poly: &[P2], skip: i64, half_window: i64) -> (P2, P2) {
    let n = poly.len() as i64;
    if n < 4 {
        return ([1.0, 0.0], [1.0, 0.0]);
    }
    let here = poly[0];
    let mut fwd_anchor = (n - 2).max(0).min(skip + 1);
    while fwd_anchor < n - 2 && !is_integer_point(poly[fwd_anchor as usize]) {
        fwd_anchor += 1;
    }
    let fwd_hi = (n - 2).min(fwd_anchor + half_window);
    let tangent_out = if fwd_hi > fwd_anchor {
        let samples = &poly[fwd_anchor as usize..=fwd_hi as usize];
        pca_unit_tangent(samples, sub(samples[samples.len() - 1], here))
    } else {
        [1.0, 0.0]
    };
    let mut bwd_anchor = (n - 2 - skip).max(1);
    while bwd_anchor > 1 && !is_integer_point(poly[bwd_anchor as usize]) {
        bwd_anchor -= 1;
    }
    let bwd_lo = (bwd_anchor - half_window).max(1);
    let tangent_in = if bwd_anchor > bwd_lo {
        let samples = &poly[bwd_lo as usize..=bwd_anchor as usize];
        pca_unit_tangent(samples, sub(samples[0], here))
    } else {
        [1.0, 0.0]
    };
    (tangent_in, tangent_out)
}

fn is_cusp(tangent_in: P2, tangent_out: P2, threshold_deg: f64) -> bool {
    let incoming = [-tangent_in[0], -tangent_in[1]];
    let cos = dot(incoming, tangent_out).clamp(-1.0, 1.0);
    cos.lacos().to_degrees() >= threshold_deg
}

fn make_participation(poly: &[P2], polyline_index: usize, point_index: usize, cfg: &Config) -> Participation {
    let n = poly.len();
    let closed = is_closed_polyline(poly);
    let is_start = point_index == 0;
    let is_end = point_index == n - 1;
    let is_endpoint_index = is_start || is_end;
    if is_endpoint_index && !closed {
        let terminal_end = if is_start { 0 } else { 1 };
        let tangent_in = terminal_tangent(poly, terminal_end, cfg.gb_terminal_tangent_window);
        return Participation {
            polyline_index,
            point_index,
            role: Role::Terminal,
            terminal_end: Some(terminal_end),
            tangent_in,
            tangent_out: None,
        };
    }
    let skip = cfg.gb_crossing_tangent_skip;
    let (tangent_in, tangent_out) = if is_endpoint_index && closed {
        closed_loop_tangents(poly, skip, cfg.gb_crossing_tangent_half_window)
    } else {
        crossing_tangents(poly, point_index, skip, cfg.gb_crossing_tangent_half_window)
    };
    let role = if is_cusp(tangent_in, tangent_out, cfg.gb_cusp_angle_threshold_deg) { Role::Cusp } else { Role::Crossing };
    Participation { polyline_index, point_index, role, terminal_end: None, tangent_in, tangent_out: Some(tangent_out) }
}

fn cluster_cross_polyline_points(polylines: &[Polyline], tol: f64) -> Vec<Vec<(usize, usize)>> {
    let mut entries: Vec<(usize, usize)> = Vec::new();
    let mut coords: Vec<P2> = Vec::new();
    for (pi, poly) in polylines.iter().enumerate() {
        for (ii, &p) in poly.iter().enumerate() {
            entries.push((pi, ii));
            coords.push(p);
        }
    }
    if coords.is_empty() {
        return vec![];
    }
    let cell = tol.max(1e-9);
    let mut grid: indexmap::IndexMap<(i64, i64), Vec<usize>> = indexmap::IndexMap::new();
    for idx in 0..coords.len() {
        let key = (py_floordiv(coords[idx][0], cell) as i64, py_floordiv(coords[idx][1], cell) as i64);
        grid.entry(key).or_default().push(idx);
    }
    let n = coords.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let tol2 = tol * tol;
    let keys: Vec<(i64, i64)> = grid.keys().cloned().collect();
    for (cx, cy) in keys {
        let idxs = grid.get(&(cx, cy)).unwrap().clone();
        let mut candidates: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(v) = grid.get(&(cx + dx, cy + dy)) {
                    candidates.extend_from_slice(v);
                }
            }
        }
        for &i in &idxs {
            let pi_i = entries[i].0;
            let (xi, yi) = (coords[i][0], coords[i][1]);
            for &j in &candidates {
                if j <= i || entries[j].0 == pi_i {
                    continue;
                }
                let dxv = xi - coords[j][0];
                let dyv = yi - coords[j][1];
                if dxv * dxv + dyv * dyv < tol2 {
                    let ra = find(&mut parent, i);
                    let rb = find(&mut parent, j);
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
    }
    let mut groups: indexmap::IndexMap<usize, Vec<usize>> = indexmap::IndexMap::new();
    for idx in 0..n {
        let r = find(&mut parent, idx);
        groups.entry(r).or_default().push(idx);
    }
    let mut out = Vec::new();
    for (_, members) in groups {
        let cluster: Vec<(usize, usize)> = members.iter().map(|&i| entries[i]).collect();
        let polys: std::collections::HashSet<usize> = cluster.iter().map(|c| c.0).collect();
        if polys.len() >= 2 {
            out.push(cluster);
        }
    }
    out
}

fn merge_clusters_by_polyline_proximity(
    clusters: Vec<Vec<(usize, usize)>>,
    polylines: &[Polyline],
    max_centroid_distance: f64,
    max_index_gap: i64,
) -> Vec<Vec<(usize, usize)>> {
    if clusters.len() <= 1 {
        return clusters;
    }
    let mut centroids: Vec<P2> = Vec::new();
    let mut ranges: Vec<indexmap::IndexMap<usize, (i64, i64)>> = Vec::new();
    for cl in &clusters {
        let xy: Vec<P2> = cl.iter().map(|&(pi, ii)| polylines[pi][ii]).collect();
        centroids.push(mean_axis0(&xy));
        let mut per_poly: indexmap::IndexMap<usize, Vec<i64>> = indexmap::IndexMap::new();
        for &(pi, ii) in cl {
            per_poly.entry(pi).or_default().push(ii as i64);
        }
        let mut r = indexmap::IndexMap::new();
        for (pi, ix) in per_poly {
            r.insert(pi, (*ix.iter().min().unwrap(), *ix.iter().max().unwrap()));
        }
        ranges.push(r);
    }
    let n = clusters.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let max_dist_sq = max_centroid_distance * max_centroid_distance;
    for i in 0..n {
        for j in i + 1..n {
            let shared: Vec<usize> = ranges[i].keys().filter(|k| ranges[j].contains_key(*k)).cloned().collect();
            if shared.len() < 2 {
                continue;
            }
            let dx = centroids[i][0] - centroids[j][0];
            let dy = centroids[i][1] - centroids[j][1];
            if dx * dx + dy * dy > max_dist_sq {
                continue;
            }
            let mut ok = true;
            for pi in shared {
                let (lo_i, hi_i) = ranges[i][&pi];
                let (lo_j, hi_j) = ranges[j][&pi];
                let gap = if lo_i > hi_j {
                    lo_i - hi_j
                } else if lo_j > hi_i {
                    lo_j - hi_i
                } else {
                    0
                };
                if gap > max_index_gap {
                    ok = false;
                    break;
                }
            }
            if ok {
                let ra = find(&mut parent, i);
                let rb = find(&mut parent, j);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut groups: indexmap::IndexMap<usize, Vec<usize>> = indexmap::IndexMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut merged = Vec::new();
    for (_, idxs) in groups {
        let mut combined = Vec::new();
        for i in idxs {
            combined.extend_from_slice(&clusters[i]);
        }
        merged.push(combined);
    }
    merged
}

pub fn build_graph(polylines: Vec<Polyline>, cfg: &Config) -> StrokeGraph {
    let mut clusters = cluster_cross_polyline_points(&polylines, cfg.gb_junction_tol);
    let merge_dist = cfg.gb_cluster_merge_centroid_distance;
    if merge_dist > 0.0 && clusters.len() > 1 {
        clusters = merge_clusters_by_polyline_proximity(clusters, &polylines, merge_dist, cfg.gb_cluster_merge_index_gap);
    }
    let mut junctions = Vec::new();
    for cluster in clusters {
        let xy: Vec<P2> = cluster.iter().map(|&(pi, ii)| polylines[pi][ii]).collect();
        let centroid = mean_axis0(&xy);
        let mut per_poly_rep: indexmap::IndexMap<usize, (usize, bool)> = indexmap::IndexMap::new();
        for &(pi, ii) in &cluster {
            let n = polylines[pi].len();
            let is_endpoint = ii == 0 || ii == n - 1;
            let dv = sub(polylines[pi][ii], centroid);
            let d2 = dv[0] * dv[0] + dv[1] * dv[1];
            match per_poly_rep.get(&pi).cloned() {
                None => {
                    per_poly_rep.insert(pi, (ii, is_endpoint));
                }
                Some((cur, cur_is_endpoint)) => {
                    let cv = sub(polylines[pi][cur], centroid);
                    let cur_d2 = cv[0] * cv[0] + cv[1] * cv[1];
                    if is_endpoint && !cur_is_endpoint {
                        per_poly_rep.insert(pi, (ii, true));
                    } else if is_endpoint == cur_is_endpoint && d2 < cur_d2 {
                        per_poly_rep.insert(pi, (ii, cur_is_endpoint));
                    }
                }
            }
        }
        if per_poly_rep.len() < 2 {
            continue;
        }
        let participants: Vec<Participation> =
            per_poly_rep.iter().map(|(&pi, &(ii, _))| make_participation(&polylines[pi], pi, ii, cfg)).collect();
        let rep_xy: Vec<P2> = participants.iter().map(|p| polylines[p.polyline_index][p.point_index]).collect();
        let location = mean_axis0(&rep_xy);
        junctions.push(Junction { location, participants });
    }
    StrokeGraph { polylines, junctions }
}
