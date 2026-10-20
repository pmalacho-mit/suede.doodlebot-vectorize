//! Segment stage: skeleton tracing, A* fusion, junction repair and post-repair
//! fusion (`segment/*.py`).

use crate::fmath::DetMath;
use crate::config::Config;
use crate::ndimage::{label, Mask, OFFS8};
use crate::numeric::{
    allclose2, dist, dot, mean_axis0, norm2, np_convolve_same, np_gradient, np_linspace, np_max, np_median, np_sum,
    py_floordiv, solve2, sub, svd_vt_n2, P2,
};
use std::collections::{BinaryHeap, HashMap};

pub type Polyline = Vec<P2>;

// ---------------------------------------------------------------------------
// trace.py

fn skeleton_neighbors(skel: &Mask, y: i64, x: i64) -> impl Iterator<Item = (i64, i64)> + '_ {
    OFFS8.iter().filter_map(move |&(dy, dx)| {
        let ny = y + dy;
        let nx = x + dx;
        if skel.get(ny, nx) {
            Some((ny, nx))
        } else {
            None
        }
    })
}

pub fn trace_skeleton(skel: &Mask) -> Vec<Polyline> {
    let (h, w) = (skel.h, skel.w);
    // degree
    let mut deg = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            if skel.at(y, x) {
                deg[y * w + x] = skeleton_neighbors(skel, y as i64, x as i64).count() as u8;
            }
        }
    }
    // node ids
    let mut junction = Mask::new(h, w, false);
    for i in 0..h * w {
        junction.data[i] = deg[i] >= 3;
    }
    let (sj, n_sj) = label(&junction, true);
    let mut node_id = vec![0i32; h * w];
    for i in 0..h * w {
        if junction.data[i] {
            node_id[i] = sj.data[i];
        }
    }
    let mut next_id = n_sj as i32 + 1;
    for i in 0..h * w {
        if deg[i] == 1 {
            node_id[i] = next_id;
            next_id += 1;
        }
    }
    let nid = |p: (i64, i64)| node_id[p.0 as usize * w + p.1 as usize];

    let mut polylines: Vec<Polyline> = Vec::new();
    let mut visited_starts: std::collections::HashSet<((i64, i64), (i64, i64))> = std::collections::HashSet::new();

    for i in 0..h * w {
        if node_id[i] == 0 {
            continue;
        }
        let sy = (i / w) as i64;
        let sx = (i % w) as i64;
        let start_pixel = (sy, sx);
        let start_id = node_id[i];
        let nbrs: Vec<(i64, i64)> = skeleton_neighbors(skel, sy, sx).collect();
        for first_step in nbrs {
            let key = (start_pixel, first_step);
            if visited_starts.contains(&key) {
                continue;
            }
            if nid(first_step) == start_id {
                continue;
            }
            let mut path = vec![start_pixel, first_step];
            let mut prev = start_pixel;
            let mut cur = first_step;
            while nid(cur) == 0 {
                let mut next_p: Option<(i64, i64)> = None;
                for cand in skeleton_neighbors(skel, cur.0, cur.1) {
                    if cand == prev {
                        continue;
                    }
                    if nid(cand) == 0 {
                        next_p = Some(cand);
                        break;
                    }
                    if next_p.is_none() {
                        next_p = Some(cand);
                    }
                }
                match next_p {
                    None => break,
                    Some(np) => {
                        prev = cur;
                        cur = np;
                        path.push(cur);
                    }
                }
            }
            visited_starts.insert(key);
            if path.len() >= 2 {
                visited_starts.insert((path[path.len() - 1], path[path.len() - 2]));
            }
            polylines.push(path.iter().map(|&(y, x)| [x as f64, y as f64]).collect());
        }
    }
    // closed loops
    let mut seen = vec![false; h * w];
    for poly in &polylines {
        for p in poly {
            let y = p[1].round() as usize;
            let x = p[0].round() as usize;
            seen[y * w + x] = true;
        }
    }
    for i in 0..h * w {
        if !(skel.data[i] && node_id[i] == 0 && !seen[i]) {
            continue;
        }
        let y = (i / w) as i64;
        let x = (i % w) as i64;
        let mut path = vec![(y, x)];
        seen[i] = true;
        let mut prev: Option<(i64, i64)> = None;
        let mut cur = (y, x);
        loop {
            let mut next_p = None;
            for nb in skeleton_neighbors(skel, cur.0, cur.1) {
                if Some(nb) != prev && !seen[nb.0 as usize * w + nb.1 as usize] {
                    next_p = Some(nb);
                    break;
                }
            }
            match next_p {
                None => break,
                Some(np) => {
                    seen[np.0 as usize * w + np.1 as usize] = true;
                    path.push(np);
                    prev = Some(cur);
                    cur = np;
                }
            }
        }
        if path.len() >= 3 {
            path.push(path[0]);
            polylines.push(path.iter().map(|&(y, x)| [x as f64, y as f64]).collect());
        }
    }
    polylines
}

pub fn polyline_length(p: &[P2]) -> f64 {
    if p.len() < 2 {
        return 0.0;
    }
    let d: Vec<f64> = (1..p.len()).map(|i| crate::numeric::dist_naive(p[i], p[i - 1])).collect();
    np_sum(&d)
}

pub fn segment_skeleton(skel: &Mask, min_length: f64) -> Vec<Polyline> {
    let polys = trace_skeleton(skel);
    if min_length > 0.0 {
        polys.into_iter().filter(|p| p.len() >= 2 && polyline_length(p) >= min_length).collect()
    } else {
        polys
    }
}

// ---------------------------------------------------------------------------
// _helpers.py

/// Union-find clustering with the cross-polyline rule (`spatial_clusters`).
pub fn spatial_clusters(coords: &[P2], tol: f64, point_poly_idx: &[usize]) -> Vec<Vec<usize>> {
    let n = coords.len();
    if n == 0 {
        return vec![];
    }
    let cell_size = tol.max(1e-9);
    let mut grid: indexmap::IndexMap<(i64, i64), Vec<usize>> = indexmap::IndexMap::new();
    for idx in 0..n {
        let c = (py_floordiv(coords[idx][0], cell_size) as i64, py_floordiv(coords[idx][1], cell_size) as i64);
        grid.entry(c).or_default().push(idx);
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let tol_sq = tol * tol;
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
            let (xi, yi) = (coords[i][0], coords[i][1]);
            let pi_i = point_poly_idx[i];
            for &j in &candidates {
                if j <= i || point_poly_idx[j] == pi_i {
                    continue;
                }
                let dx_ = xi - coords[j][0];
                let dy_ = yi - coords[j][1];
                if dx_ * dx_ + dy_ * dy_ < tol_sq {
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
    groups.into_values().collect()
}

/// PCA principal direction (Vt[0]) and perpendicular (Vt[1]) of centered points.
fn pca(pts: &[P2]) -> (P2, P2, P2) {
    let centroid = mean_axis0(pts);
    let centered: Vec<P2> = pts.iter().map(|&p| sub(p, centroid)).collect();
    let vt = svd_vt_n2(&centered);
    (centroid, vt[0], vt[1])
}

pub fn stable_endpoint_tangent_and_curvature(poly: &[P2], end_side: usize, skip: usize, sample: usize) -> (P2, f64) {
    let n = poly.len();
    if n < 2 {
        return ([1.0, 0.0], 0.0);
    }
    let (lo, hi) = if end_side == 0 {
        let lo = skip.min(n.saturating_sub(2));
        (lo, (lo + sample).min(n))
    } else {
        let hi = (n as i64 - skip as i64).max(2) as usize;
        (hi.saturating_sub(sample), hi)
    };
    let pts = &poly[lo..hi];
    if pts.len() < 2 {
        let d = if end_side == 0 { sub(poly[n - 1], poly[0]) } else { sub(poly[0], poly[n - 1]) };
        let nn = norm2(d);
        return (if nn > 1e-9 { [d[0] / nn, d[1] / nn] } else { [1.0, 0.0] }, 0.0);
    }
    let (centroid, mut tangent, perp) = pca(pts);
    let endpoint = if end_side == 0 { poly[0] } else { poly[n - 1] };
    let r = sub(endpoint, centroid);
    if dot(tangent, r) < 0.0 {
        tangent = [-tangent[0], -tangent[1]];
    }
    if pts.len() < 3 {
        return (tangent, 0.0);
    }
    let k = pts.len();
    let max_perp = pts.iter().map(|&p| crate::blas::gemv_row(sub(p, centroid), perp, k).abs()).fold(f64::NEG_INFINITY, f64::max);
    let chord = dist(pts[pts.len() - 1], pts[0]);
    if chord < 1.0 {
        return (tangent, 0.0);
    }
    (tangent, 8.0 * max_perp / (chord * chord))
}

// ---------------------------------------------------------------------------
// fusion.py

#[derive(Clone, Copy)]
struct HeapItem {
    f: f64,
    counter: usize,
    node: (i64, i64),
}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.counter == o.counter
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // min-heap on (f, counter)
        o.f.partial_cmp(&self.f).unwrap().then(o.counter.cmp(&self.counter))
    }
}

fn shortest_path_through_binary(
    binary: &Mask,
    start: (i64, i64),
    goal: (i64, i64),
    max_cost: f64,
) -> Option<Vec<(i64, i64)>> {
    if !binary.get(start.0, start.1) || !binary.get(goal.0, goal.1) {
        return None;
    }
    if start == goal {
        return Some(vec![start]);
    }
    let sqrt2 = 2.0f64.sqrt();
    let moves: [(i64, i64, f64); 8] = [
        (-1, -1, sqrt2),
        (-1, 0, 1.0),
        (-1, 1, sqrt2),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, sqrt2),
        (1, 0, 1.0),
        (1, 1, sqrt2),
    ];
    let octile = |p: (i64, i64)| -> f64 {
        let dy = (p.0 - goal.0).abs() as f64;
        let dx = (p.1 - goal.1).abs() as f64;
        if dy > dx {
            dy + (sqrt2 - 1.0) * dx
        } else {
            dx + (sqrt2 - 1.0) * dy
        }
    };
    // Any reachable node lies within `max_cost` Chebyshev distance of the start:
    // use a dense local window instead of hash maps.
    let r = max_cost.ceil() as i64 + 1;
    let (y0, x0) = (start.0 - r, start.1 - r);
    let side = (2 * r + 1) as usize;
    let idx = |p: (i64, i64)| -> usize { ((p.0 - y0) as usize) * side + (p.1 - x0) as usize };
    let mut g_score = vec![f64::INFINITY; side * side];
    let mut parent: Vec<u32> = vec![u32::MAX; side * side];
    let mut heap = BinaryHeap::new();
    let mut counter = 0usize;
    g_score[idx(start)] = 0.0;
    heap.push(HeapItem { f: octile(start), counter, node: start });
    while let Some(item) = heap.pop() {
        let cur = item.node;
        if cur == goal {
            let mut path = vec![goal];
            let mut c = idx(goal);
            while parent[c] != u32::MAX {
                let pi = parent[c] as usize;
                let p = (y0 + (pi / side) as i64, x0 + (pi % side) as i64);
                path.push(p);
                c = pi;
            }
            path.reverse();
            return Some(path);
        }
        let cur_g = g_score[idx(cur)];
        for &(dy, dx, step) in &moves {
            let ny = cur.0 + dy;
            let nx = cur.1 + dx;
            if !binary.get(ny, nx) {
                continue;
            }
            let tentative = cur_g + step;
            if tentative > max_cost {
                continue;
            }
            let nbr = (ny, nx);
            let ni = idx(nbr);
            if tentative >= g_score[ni] {
                continue;
            }
            g_score[ni] = tentative;
            parent[ni] = idx(cur) as u32;
            counter += 1;
            heap.push(HeapItem { f: tentative + octile(nbr), counter, node: nbr });
        }
    }
    None
}

fn local_tangent_and_curvature(poly: &[P2], at_end: bool, lookback: usize) -> (P2, f64) {
    let n = lookback.min(poly.len());
    let len = poly.len();
    if n < 3 {
        let d = if at_end { sub(poly[len - 1], poly[0]) } else { sub(poly[0], poly[len - 1]) };
        let nn = norm2(d);
        return (if nn > 1e-9 { [d[0] / nn, d[1] / nn] } else { [0.0, 0.0] }, 0.0);
    }
    let pts = if at_end { &poly[len - n..] } else { &poly[..n] };
    let (centroid, mut tangent, perp) = pca(pts);
    let r = if at_end { sub(pts[pts.len() - 1], pts[0]) } else { sub(pts[0], pts[pts.len() - 1]) };
    if dot(tangent, r) < 0.0 {
        tangent = [-tangent[0], -tangent[1]];
    }
    let k = pts.len();
    let max_perp = pts.iter().map(|&p| crate::blas::gemv_row(sub(p, centroid), perp, k).abs()).fold(f64::NEG_INFINITY, f64::max);
    let chord = dist(pts[pts.len() - 1], pts[0]);
    if chord < 1.0 {
        return (tangent, 0.0);
    }
    (tangent, 8.0 * max_perp / (chord * chord))
}

struct Candidate {
    seg_a: usize,
    end_a: usize,
    seg_b: usize,
    end_b: usize,
    path: Vec<P2>,
    tangent_score: f64,
    score: f64,
}

pub fn fuse_segments(segments: &[Polyline], binary: &Mask, cfg: &Config) -> Vec<Polyline> {
    let max_path_length = cfg.fuse_max_path_length;
    let mut endpoints: Vec<(usize, usize, P2)> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if seg.len() < 2 {
            continue;
        }
        endpoints.push((i, 0, seg[0]));
        endpoints.push((i, 1, seg[seg.len() - 1]));
    }
    let mut cands: Vec<Candidate> = Vec::new();
    // cache tangents per endpoint
    let mut tan_cache: HashMap<(usize, usize), (P2, f64)> = HashMap::new();
    for ai in 0..endpoints.len() {
        let (seg_a, end_a, pt_a) = endpoints[ai];
        for bi in ai + 1..endpoints.len() {
            let (seg_b, end_b, pt_b) = endpoints[bi];
            if seg_a == seg_b {
                continue;
            }
            let d_eucl = dist(pt_a, pt_b);
            if d_eucl > 2.0 * max_path_length as f64 {
                continue;
            }
            let (ya, xa) = (crate::numeric::py_round(pt_a[1]) as i64, crate::numeric::py_round(pt_a[0]) as i64);
            let (yb, xb) = (crate::numeric::py_round(pt_b[1]) as i64, crate::numeric::py_round(pt_b[0]) as i64);
            let path_yx = match shortest_path_through_binary(binary, (ya, xa), (yb, xb), max_path_length as f64) {
                Some(p) => p,
                None => continue,
            };
            let path_xy: Vec<P2> = path_yx.iter().map(|&(y, x)| [x as f64, y as f64]).collect();
            let path_length = path_yx.len() as f64;
            let (tan_a, curv_a) = *tan_cache
                .entry((seg_a, end_a))
                .or_insert_with(|| local_tangent_and_curvature(&segments[seg_a], end_a == 1, cfg.fuse_lookback));
            let (tan_b, curv_b) = *tan_cache
                .entry((seg_b, end_b))
                .or_insert_with(|| local_tangent_and_curvature(&segments[seg_b], end_b == 1, cfg.fuse_lookback));
            let tangent_score = -dot(tan_a, tan_b);
            let curvature_diff = (curv_a - curv_b).abs();
            let score = tangent_score - cfg.fuse_gap_penalty * path_length - cfg.fuse_curvature_penalty * curvature_diff;
            cands.push(Candidate { seg_a, end_a, seg_b, end_b, path: path_xy, tangent_score, score });
        }
    }
    let mut cands: Vec<Candidate> = cands.into_iter().filter(|c| c.tangent_score >= cfg.fuse_min_tangent_score).collect();
    cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let n = segments.len();
    let mut used: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    // link[seg][end] = (other_seg, other_end, path)
    let mut link: Vec<[Option<(usize, usize, Vec<P2>)>; 2]> = (0..n).map(|_| [None, None]).collect();
    for c in cands {
        let ka = (c.seg_a, c.end_a);
        let kb = (c.seg_b, c.end_b);
        if used.contains(&ka) || used.contains(&kb) {
            continue;
        }
        used.insert(ka);
        used.insert(kb);
        let mut rev = c.path.clone();
        rev.reverse();
        link[c.seg_a][c.end_a] = Some((c.seg_b, c.end_b, c.path));
        link[c.seg_b][c.end_b] = Some((c.seg_a, c.end_a, rev));
    }
    let mut visited = vec![false; n];
    let mut fused: Vec<Polyline> = Vec::new();
    let walk = |start_seg: usize, start_end: usize, visited: &mut Vec<bool>| -> Polyline {
        let mut chain: Polyline = Vec::new();
        let (mut cur_seg, mut cur_end) = (start_seg, start_end);
        while !visited[cur_seg] {
            visited[cur_seg] = true;
            let poly = &segments[cur_seg];
            let pts: Vec<P2> = if cur_end == 0 { poly.clone() } else { poly.iter().rev().cloned().collect() };
            if !chain.is_empty() && allclose2(*chain.last().unwrap(), pts[0], 0.5) {
                chain.extend_from_slice(&pts[1..]);
            } else {
                chain.extend_from_slice(&pts);
            }
            let exit_end = 1 - cur_end;
            let li = match &link[cur_seg][exit_end] {
                None => break,
                Some(l) => l,
            };
            let (next_seg, next_end, conn) = (li.0, li.1, &li.2);
            if visited[next_seg] {
                break;
            }
            if conn.len() > 2 {
                chain.extend_from_slice(&conn[1..conn.len() - 1]);
            }
            cur_seg = next_seg;
            cur_end = next_end;
        }
        chain
    };
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let start_end = if link[i][0].is_none() {
            Some(0)
        } else if link[i][1].is_none() {
            Some(1)
        } else {
            None
        };
        if let Some(se) = start_end {
            fused.push(walk(i, se, &mut visited));
        }
    }
    for i in 0..n {
        if visited[i] {
            continue;
        }
        fused.push(walk(i, 0, &mut visited));
    }
    fused
}

pub fn fuse_post_repair(polylines: &[Polyline], cfg: &Config) -> Vec<Polyline> {
    let n = polylines.len();
    if n == 0 {
        return vec![];
    }
    let mut meta: Vec<(usize, usize)> = Vec::new();
    let mut coords: Vec<P2> = Vec::new();
    for (pi, poly) in polylines.iter().enumerate() {
        if poly.len() < 2 {
            continue;
        }
        meta.push((pi, 0));
        coords.push(poly[0]);
        meta.push((pi, 1));
        coords.push(poly[poly.len() - 1]);
    }
    if coords.is_empty() {
        return polylines.to_vec();
    }
    let point_poly_idx: Vec<usize> = meta.iter().map(|m| m.0).collect();
    let clusters = spatial_clusters(&coords, cfg.prf_junction_tol, &point_poly_idx);
    struct C {
        seg_a: usize,
        end_a: usize,
        seg_b: usize,
        end_b: usize,
        score: f64,
    }
    let mut candidates: Vec<C> = Vec::new();
    for cl in clusters {
        let members: Vec<(usize, usize)> = cl.iter().map(|&i| meta[i]).collect();
        let distinct: std::collections::HashSet<usize> = members.iter().map(|m| m.0).collect();
        if distinct.len() < 2 {
            continue;
        }
        let tans: Vec<(usize, usize, P2, f64)> = members
            .iter()
            .map(|&(pi, end)| {
                let (t, c) =
                    stable_endpoint_tangent_and_curvature(&polylines[pi], end, cfg.prf_tangent_skip, cfg.prf_tangent_sample);
                (pi, end, t, c)
            })
            .collect();
        let m = tans.len();
        for i in 0..m {
            let (pa, ea, ta, ca) = tans[i];
            for j in i + 1..m {
                let (pb, eb, tb, cb) = tans[j];
                if pa == pb {
                    continue;
                }
                let tan_score = -dot(ta, tb);
                if tan_score < cfg.prf_min_tangent_score {
                    continue;
                }
                let score = tan_score - cfg.prf_curvature_penalty * (ca - cb).abs();
                candidates.push(C { seg_a: pa, end_a: ea, seg_b: pb, end_b: eb, score });
            }
        }
    }
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut used: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut link: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; n];
    for c in candidates {
        let ka = (c.seg_a, c.end_a);
        let kb = (c.seg_b, c.end_b);
        if used.contains(&ka) || used.contains(&kb) {
            continue;
        }
        used.insert(ka);
        used.insert(kb);
        link[c.seg_a][c.end_a] = Some((c.seg_b, c.end_b));
        link[c.seg_b][c.end_b] = Some((c.seg_a, c.end_a));
    }
    let mut visited = vec![false; n];
    let mut fused: Vec<Polyline> = Vec::new();
    let walk = |start_seg: usize, start_end: usize, visited: &mut Vec<bool>| -> Polyline {
        let mut chain: Polyline = Vec::new();
        let (mut cur_seg, mut cur_end) = (start_seg, start_end);
        while !visited[cur_seg] {
            visited[cur_seg] = true;
            let poly = &polylines[cur_seg];
            let pts: Vec<P2> = if cur_end == 0 { poly.clone() } else { poly.iter().rev().cloned().collect() };
            if !chain.is_empty() && allclose2(*chain.last().unwrap(), pts[0], 0.5) {
                chain.extend_from_slice(&pts[1..]);
            } else {
                chain.extend_from_slice(&pts);
            }
            let exit_end = 1 - cur_end;
            let (next_seg, next_end) = match link[cur_seg][exit_end] {
                None => break,
                Some(l) => l,
            };
            if visited[next_seg] {
                break;
            }
            cur_seg = next_seg;
            cur_end = next_end;
        }
        chain
    };
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let start_end = if link[i][0].is_none() {
            Some(0)
        } else if link[i][1].is_none() {
            Some(1)
        } else {
            None
        };
        if let Some(se) = start_end {
            fused.push(walk(i, se, &mut visited));
        }
    }
    for i in 0..n {
        if visited[i] {
            continue;
        }
        fused.push(walk(i, 0, &mut visited));
    }
    fused
}

// ---------------------------------------------------------------------------
// repair.py

type Arm = (P2, P2); // (anchor, unit tangent)

fn build_approaches(poly: &[P2], rs: i64, re: i64, stable_skip: i64, stable_sample: i64) -> Vec<Arm> {
    let n = poly.len() as i64;
    let mut arms = Vec::new();
    if rs > 0 {
        let inner_idx = rs - 1;
        let mut sample_near = inner_idx - stable_skip;
        let mut sample_far = inner_idx - stable_skip - stable_sample + 1;
        sample_near = sample_near.max(0);
        sample_far = sample_far.max(0);
        if sample_near < sample_far {
            std::mem::swap(&mut sample_near, &mut sample_far);
        }
        if sample_near - sample_far < 1 {
            sample_far = 0;
            sample_near = inner_idx;
        }
        if sample_near > sample_far {
            let samples = &poly[sample_far as usize..=sample_near as usize];
            let anchor = mean_axis0(samples);
            let v = sub(samples[0], samples[samples.len() - 1]);
            let nn = norm2(v);
            if nn > 1e-9 {
                arms.push((anchor, [v[0] / nn, v[1] / nn]));
            }
        }
    }
    if re < n - 1 {
        let inner_idx = re + 1;
        let mut sample_near = inner_idx + stable_skip;
        let mut sample_far = inner_idx + stable_skip + stable_sample - 1;
        sample_near = sample_near.min(n - 1);
        sample_far = sample_far.min(n - 1);
        if sample_far < sample_near {
            std::mem::swap(&mut sample_near, &mut sample_far);
        }
        if sample_far - sample_near < 1 {
            sample_near = inner_idx;
            sample_far = n - 1;
        }
        if sample_far > sample_near {
            let samples = &poly[sample_near as usize..=sample_far as usize];
            let anchor = mean_axis0(samples);
            let v = sub(samples[samples.len() - 1], samples[0]);
            let nn = norm2(v);
            if nn > 1e-9 {
                arms.push((anchor, [v[0] / nn, v[1] / nn]));
            }
        }
    }
    arms
}

fn tangent_spread_deg(tangents: &[P2]) -> f64 {
    if tangents.len() < 2 {
        return 180.0;
    }
    let pi = std::f64::consts::PI;
    let mut angles: Vec<f64> = tangents.iter().map(|t| crate::numeric::py_mod(t[1].latan2(t[0]), pi)).collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut gaps: Vec<f64> = (1..angles.len()).map(|i| angles[i] - angles[i - 1]).collect();
    gaps.push(pi - angles[angles.len() - 1] + angles[0]);
    let max_gap = np_max(&gaps);
    (pi - max_gap).to_degrees()
}

fn solve_junction_point(anchors: &[P2], tangents: &[P2], fallback: P2) -> P2 {
    if anchors.is_empty() {
        return fallback;
    }
    if anchors.len() == 1 {
        return anchors[0];
    }
    let mut a = [[0.0f64; 2]; 2];
    let mut b = [0.0f64; 2];
    for (an, t) in anchors.iter().zip(tangents) {
        // M = I - t t^T
        let m = [[1.0 - t[0] * t[0], 0.0 - t[0] * t[1]], [0.0 - t[1] * t[0], 1.0 - t[1] * t[1]]];
        a[0][0] += m[0][0];
        a[0][1] += m[0][1];
        a[1][0] += m[1][0];
        a[1][1] += m[1][1];
        b[0] += crate::blas::gemv_row([m[0][0], m[0][1]], *an, 2);
        b[1] += crate::blas::gemv_row([m[1][0], m[1][1]], *an, 2);
    }
    match solve2(a, b) {
        Some(x) => x,
        None => fallback,
    }
}

fn polyline_d2_mag(poly: &[P2]) -> Vec<f64> {
    let n = poly.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let xs: Vec<f64> = poly.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = poly.iter().map(|p| p[1]).collect();
    let dx = np_gradient(&xs);
    let dy = np_gradient(&ys);
    let ddx = np_gradient(&dx);
    let ddy = np_gradient(&dy);
    let mag: Vec<f64> = (0..n).map(|i| (ddx[i] * ddx[i] + ddy[i] * ddy[i]).sqrt()).collect();
    let k = 3usize.min(n);
    let kernel = vec![1.0 / k as f64; k];
    np_convolve_same(&mag, &kernel)
}

fn kink_present(d2: &[f64], rs: i64, re: i64, context_window: i64, spike_ratio: f64) -> bool {
    let n = d2.len() as i64;
    if n == 0 || rs > re {
        return false;
    }
    let search_pad = 4;
    let rs_c = rs.max(0);
    let re_c = re.min(n - 1);
    let search_start = (rs_c - search_pad).max(0);
    let search_end = (re_c + search_pad).min(n - 1);
    let in_search = &d2[search_start as usize..=search_end as usize];
    if in_search.is_empty() {
        return false;
    }
    let base_start = (search_start - context_window).max(0);
    let base_end = (search_end + 1 + context_window).min(n);
    let mut outside: Vec<f64> = Vec::new();
    let mut any = false;
    if base_start < search_start {
        outside.extend_from_slice(&d2[base_start as usize..search_start as usize]);
        any = true;
    }
    if search_end + 1 < base_end {
        outside.extend_from_slice(&d2[(search_end + 1) as usize..base_end as usize]);
        any = true;
    }
    if !any {
        return true;
    }
    if outside.len() < 2 {
        return true;
    }
    let baseline = np_median(&outside).max(1e-3);
    let spike = np_max(in_search);
    spike >= spike_ratio * baseline
}

fn extend_region_by_d2(d2: &[f64], rs: i64, re: i64, max_extend: i64, baseline_factor: f64) -> (i64, i64) {
    let n = d2.len() as i64;
    if n == 0 || rs > re {
        return (rs, re);
    }
    let rs_c = rs.max(0);
    let re_c = re.min(n - 1);
    let far_left_end = (rs_c - max_extend).max(0);
    let far_right_start = (re_c + 1 + max_extend).min(n);
    let context = 8;
    let far_left_start = (far_left_end - context).max(0);
    let far_right_end = (far_right_start + context).min(n);
    let mut outside: Vec<f64> = Vec::new();
    let mut any = false;
    if far_left_start < far_left_end {
        outside.extend_from_slice(&d2[far_left_start as usize..far_left_end as usize]);
        any = true;
    }
    if far_right_start < far_right_end {
        outside.extend_from_slice(&d2[far_right_start as usize..far_right_end as usize]);
        any = true;
    }
    if !any || outside.len() < 2 {
        return (rs_c, re_c);
    }
    let baseline = np_median(&outside).max(1e-3);
    let threshold = baseline * baseline_factor;
    let mut rs_ext = rs_c;
    for off in 1..=max_extend {
        let idx = rs_c - off;
        if idx < 0 {
            break;
        }
        if d2[idx as usize] > threshold {
            rs_ext = idx;
        } else {
            break;
        }
    }
    let mut re_ext = re_c;
    for off in 1..=max_extend {
        let idx = re_c + off;
        if idx >= n {
            break;
        }
        if d2[idx as usize] > threshold {
            re_ext = idx;
        } else {
            break;
        }
    }
    (rs_ext, re_ext)
}

fn interpolate(start: P2, end: P2, max_spacing: f64) -> Vec<P2> {
    let d = dist(end, start);
    if d <= max_spacing || max_spacing <= 0.0 {
        return vec![];
    }
    let n_segments = (d / max_spacing).ceil() as usize;
    let t = np_linspace(0.0, 1.0, n_segments + 1);
    let dv = sub(end, start);
    t[1..t.len() - 1].iter().map(|&tt| [start[0] + tt * dv[0], start[1] + tt * dv[1]]).collect()
}

fn merge_cascade_repairs(repairs: &[(i64, i64, P2)], cascade_gap: i64, max_jp: f64) -> Vec<(i64, i64, P2)> {
    if repairs.len() <= 1 {
        return repairs.to_vec();
    }
    let mut rs = repairs.to_vec();
    rs.sort_by_key(|r| r.0);
    let mut merged = vec![rs[0]];
    for &(js, je, jp) in &rs[1..] {
        let (ljs, lje, ljp) = *merged.last().unwrap();
        let close_in_index = js - lje <= cascade_gap;
        let close_in_space = dist(jp, ljp) <= max_jp;
        if close_in_index && close_in_space {
            let new_jp = [(ljp[0] + jp[0]) / 2.0, (ljp[1] + jp[1]) / 2.0];
            *merged.last_mut().unwrap() = (ljs, lje.max(je), new_jp);
        } else {
            merged.push((js, je, jp));
        }
    }
    merged
}

fn apply_repairs(poly: &[P2], repairs: &[(i64, i64, P2)], interp_max_spacing: f64) -> Polyline {
    if repairs.is_empty() {
        return poly.to_vec();
    }
    let mut rs = repairs.to_vec();
    rs.sort_by_key(|r| r.0);
    let mut pieces: Vec<P2> = Vec::new();
    let mut cursor: i64 = 0;
    let mut last_emitted: Option<P2> = None;
    let push = |arr: &[P2], pieces: &mut Vec<P2>, last_emitted: &mut Option<P2>| {
        if arr.is_empty() {
            return;
        }
        if let Some(le) = *last_emitted {
            let bridge = interpolate(le, arr[0], interp_max_spacing);
            pieces.extend(bridge);
        }
        pieces.extend_from_slice(arr);
        *last_emitted = Some(arr[arr.len() - 1]);
    };
    for &(js, je, jp) in &rs {
        if js > cursor {
            push(&poly[cursor as usize..js as usize], &mut pieces, &mut last_emitted);
        }
        push(&[jp], &mut pieces, &mut last_emitted);
        cursor = je + 1;
    }
    if (cursor as usize) < poly.len() {
        push(&poly[cursor as usize..], &mut pieces, &mut last_emitted);
    }
    pieces
}

pub fn repair_junctions(polylines: &[Polyline], cfg: &Config) -> Vec<Polyline> {
    if polylines.is_empty() {
        return vec![];
    }
    let mut meta: Vec<(usize, usize)> = Vec::new();
    let mut coords: Vec<P2> = Vec::new();
    for (pi, poly) in polylines.iter().enumerate() {
        for (ii, &p) in poly.iter().enumerate() {
            meta.push((pi, ii));
            coords.push(p);
        }
    }
    if coords.is_empty() {
        return polylines.to_vec();
    }
    let point_poly_idx: Vec<usize> = meta.iter().map(|m| m.0).collect();
    let raw_clusters = spatial_clusters(&coords, cfg.rep_junction_tol, &point_poly_idx);
    let clusters: Vec<Vec<usize>> = raw_clusters
        .into_iter()
        .filter(|c| {
            let s: std::collections::HashSet<usize> = c.iter().map(|&i| meta[i].0).collect();
            s.len() >= 2
        })
        .collect();
    let mut per_poly_repairs: Vec<Vec<(i64, i64, P2)>> = vec![Vec::new(); polylines.len()];
    let d2_profiles: Vec<Vec<f64>> = polylines.iter().map(|p| polyline_d2_mag(p)).collect();
    for cluster in &clusters {
        let mut poly_to_indices: indexmap::IndexMap<usize, Vec<i64>> = indexmap::IndexMap::new();
        for &gi in cluster {
            let (pi, ii) = meta[gi];
            poly_to_indices.entry(pi).or_default().push(ii as i64);
        }
        let mut too_long = false;
        for (_, idxs) in &poly_to_indices {
            let mx = *idxs.iter().max().unwrap();
            let mn = *idxs.iter().min().unwrap();
            if mx - mn + 1 > cfg.rep_max_junction_region_length {
                too_long = true;
                break;
            }
        }
        if too_long {
            continue;
        }
        let mut n_kinky = 0;
        for (&pi, idxs) in &poly_to_indices {
            let rs = *idxs.iter().min().unwrap();
            let re = *idxs.iter().max().unwrap();
            if kink_present(&d2_profiles[pi], rs, re, cfg.rep_curvature_context_window, cfg.rep_min_curvature_spike_ratio)
            {
                n_kinky += 1;
            }
        }
        if n_kinky < 1 {
            continue;
        }
        let mut extended: indexmap::IndexMap<usize, (i64, i64)> = indexmap::IndexMap::new();
        for (&pi, idxs) in &poly_to_indices {
            let rs = *idxs.iter().min().unwrap();
            let re = *idxs.iter().max().unwrap();
            extended.insert(pi, extend_region_by_d2(&d2_profiles[pi], rs, re, cfg.rep_curvature_context_window, 1.5));
        }
        let mut all_anchors: Vec<P2> = Vec::new();
        let mut all_tangents: Vec<P2> = Vec::new();
        let mut poly_to_region: indexmap::IndexMap<usize, (i64, i64)> = indexmap::IndexMap::new();
        for (&pi, _) in &poly_to_indices {
            let (rs_ext, re_ext) = extended[&pi];
            let arms = build_approaches(&polylines[pi], rs_ext, re_ext, cfg.rep_stable_skip, cfg.rep_stable_sample);
            if arms.is_empty() {
                continue;
            }
            for (anchor, tangent) in arms {
                all_anchors.push(anchor);
                all_tangents.push(tangent);
            }
            poly_to_region.insert(pi, (rs_ext, re_ext));
        }
        if poly_to_region.len() < 2 || all_anchors.len() < 2 {
            continue;
        }
        let spread = tangent_spread_deg(&all_tangents);
        if spread < cfg.rep_min_tangent_spread_deg {
            continue;
        }
        let cpts: Vec<P2> = cluster.iter().map(|&i| coords[i]).collect();
        let cluster_centroid = mean_axis0(&cpts);
        let mut jp = solve_junction_point(&all_anchors, &all_tangents, cluster_centroid);
        let max_dev = 5.0 * cfg.rep_junction_tol;
        if dist(jp, cluster_centroid) > max_dev {
            jp = cluster_centroid;
        }
        for (&pi, &(rs, re)) in &poly_to_region {
            per_poly_repairs[pi].push((rs, re, jp));
        }
    }
    let mut repaired = Vec::new();
    for (pi, poly) in polylines.iter().enumerate() {
        let raw = &per_poly_repairs[pi];
        let merged = if raw.len() >= 2 {
            merge_cascade_repairs(raw, cfg.rep_cascade_gap, cfg.rep_cascade_max_jp_distance)
        } else {
            raw.clone()
        };
        let new_poly = apply_repairs(poly, &merged, cfg.rep_interp_max_spacing);
        if new_poly.len() >= cfg.rep_min_output_polyline_length {
            repaired.push(new_poly);
        }
    }
    repaired
}

/// Full segment stage: returns (raw segments, fused_post_repair polylines).
pub fn segment_stage(skel: &Mask, binary: &Mask, cfg: &Config) -> (Vec<Polyline>, Vec<Polyline>) {
    let segments = segment_skeleton(skel, cfg.seg_min_length);
    let fused = fuse_segments(&segments, binary, cfg);
    let repaired = repair_junctions(&fused, cfg);
    let post = fuse_post_repair(&repaired, cfg);
    (segments, post)
}
