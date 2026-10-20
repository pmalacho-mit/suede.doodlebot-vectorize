//! Low-geometry vectorizer (`vectorize/low_geometry/__init__.py` + `solve.py`).

use crate::fmath::DetMath;
pub mod beautify;
pub mod fitting;
pub mod minpack;
pub mod nxgraph;
pub mod primitives;
pub mod routing;
pub mod solve;

use crate::commands::Command;
use crate::graph::{Role, StrokeGraph};
use crate::numeric::{dist, dot, np_linspace_int, P2};
use fitting::{fit_polyline, fuse_chain, is_near_closed_polyline, ChainPiece, FitParams};
use primitives::*;
use solve::{solve_once, SoftConstraints, Weights};

pub struct FittedSegment {
    pub polyline_index: usize,
    pub pieces: Vec<ChainPiece>,
    pub primitive_ids: Vec<usize>,
    pub source_points: Vec<Vec<P2>>,
}

struct FitConfig {
    line_tol: f64,
    arc_tol: f64,
    min_len: usize,
    smooth_junction_deg_threshold: f64,
    source_points_per_primitive: usize,
    polyline_subsample_cap: usize,
}
impl Default for FitConfig {
    fn default() -> Self {
        FitConfig {
            line_tol: 0.005,
            arc_tol: 0.012,
            min_len: 5,
            smooth_junction_deg_threshold: 25.0,
            source_points_per_primitive: 30,
            polyline_subsample_cap: 200,
        }
    }
}

fn build_chains(graph: &StrokeGraph, cfg: &FitConfig) -> Vec<FittedSegment> {
    let mut out = Vec::new();
    let fp = FitParams { line_tol: cfg.line_tol, arc_tol: cfg.arc_tol, min_len: cfg.min_len, circle_rms_rel: 0.06 };
    for (pi, raw) in graph.polylines.iter().enumerate() {
        if raw.len() < 2 {
            continue;
        }
        let cap_poly = 2usize.max(cfg.polyline_subsample_cap);
        let poly: Vec<P2> = if raw.len() > cap_poly {
            np_linspace_int(0.0, (raw.len() - 1) as f64, cap_poly).into_iter().map(|i| raw[i]).collect()
        } else {
            raw.clone()
        };
        let chain = fit_polyline(&poly, &fp);
        if chain.is_empty() {
            continue;
        }
        let cap = 2usize.max(cfg.source_points_per_primitive);
        let mut src = Vec::new();
        for c in &chain {
            let piece = &poly[c.start_idx..c.end_idx];
            if piece.len() <= cap {
                src.push(piece.to_vec());
            } else {
                let idx = np_linspace_int(0.0, (piece.len() - 1) as f64, cap);
                src.push(idx.into_iter().map(|i| piece[i]).collect());
            }
        }
        out.push(FittedSegment { polyline_index: pi, pieces: chain, primitive_ids: vec![], source_points: src });
    }
    out
}

fn assign_global_ids(segments: &mut [FittedSegment]) -> Vec<Prim> {
    let mut prims = Vec::new();
    for seg in segments.iter_mut() {
        seg.primitive_ids.clear();
        for piece in &seg.pieces {
            seg.primitive_ids.push(prims.len());
            prims.push(piece.primitive.clone());
        }
    }
    prims
}

fn chain_endpoint(seg: &FittedSegment, start: bool) -> (usize, bool) {
    if start {
        (seg.primitive_ids[0], true)
    } else {
        (*seg.primitive_ids.last().unwrap(), false)
    }
}

fn polyline_endpoint_to_chain_end(poly_n: usize, point_index: usize) -> Option<bool> {
    if point_index == 0 {
        Some(true)
    } else if point_index == poly_n - 1 {
        Some(false)
    } else {
        None
    }
}

fn resolve_host_subprimitive(junction_xy: P2, seg: &FittedSegment) -> usize {
    let mut best_pid = seg.primitive_ids[0];
    let mut best_dist = f64::INFINITY;
    for (k, src) in seg.source_points.iter().enumerate() {
        let d = src.iter().map(|&p| dist(p, junction_xy)).fold(f64::INFINITY, f64::min);
        if d < best_dist {
            best_dist = d;
            best_pid = seg.primitive_ids[k];
        }
    }
    best_pid
}

fn build_junction_constraints(graph: &StrokeGraph, segments: &[FittedSegment], smooth_junction_deg: f64, prims: &[Prim]) -> SoftConstraints {
    let mut seg_by_poly: std::collections::HashMap<usize, &FittedSegment> = std::collections::HashMap::new();
    for s in segments {
        // (a chain can come back empty from fuse_chain when every piece was
        // degenerate; the reference would raise there, we simply skip it)
        if !s.primitive_ids.is_empty() {
            seg_by_poly.insert(s.polyline_index, s);
        }
    }
    let mut out = SoftConstraints::default();
    let is_circle = |pid: usize| pid < prims.len() && prims[pid].is_circle();
    for j in &graph.junctions {
        let live: Vec<&crate::graph::Participation> =
            j.participants.iter().filter(|p| seg_by_poly.contains_key(&p.polyline_index)).collect();
        if live.len() < 2 {
            continue;
        }
        let terminals: Vec<&crate::graph::Participation> = live.iter().cloned().filter(|p| p.role == Role::Terminal).collect();
        let interiors: Vec<&crate::graph::Participation> = live.iter().cloned().filter(|p| p.role != Role::Terminal).collect();
        if terminals.len() >= 2 {
            let base = terminals[0];
            let base_seg = seg_by_poly[&base.polyline_index];
            if let Some(base_end) = polyline_endpoint_to_chain_end(graph.polylines[base.polyline_index].len(), base.point_index) {
                let base_ep = chain_endpoint(base_seg, base_end);
                let base_is_circle = is_circle(base_ep.0);
                for p in &terminals[1..] {
                    let seg = seg_by_poly[&p.polyline_index];
                    let end = match polyline_endpoint_to_chain_end(graph.polylines[p.polyline_index].len(), p.point_index) {
                        Some(e) => e,
                        None => continue,
                    };
                    let ep = chain_endpoint(seg, end);
                    if base_is_circle {
                        out.on_curve.push((ep, base_ep.0));
                    } else if is_circle(ep.0) {
                        out.on_curve.push((base_ep, ep.0));
                    } else {
                        out.coincide.push((base_ep, ep));
                    }
                }
            }
        }
        if !terminals.is_empty() && !interiors.is_empty() {
            let host_part = interiors[0];
            let host_seg = seg_by_poly[&host_part.polyline_index];
            let host_id = resolve_host_subprimitive(j.location, host_seg);
            for term in &terminals {
                let seg = seg_by_poly[&term.polyline_index];
                let end = match polyline_endpoint_to_chain_end(graph.polylines[term.polyline_index].len(), term.point_index) {
                    Some(e) => e,
                    None => continue,
                };
                let ep = chain_endpoint(seg, end);
                out.on_curve.push((ep, host_id));
            }
        }
        if terminals.len() == 2 && interiors.is_empty() {
            let a = terminals[0];
            let b = terminals[1];
            let cos = (-dot(a.tangent_in, b.tangent_in)).clamp(-1.0, 1.0);
            let defl = cos.lacos().to_degrees();
            if defl < smooth_junction_deg {
                let seg_a = seg_by_poly[&a.polyline_index];
                let seg_b = seg_by_poly[&b.polyline_index];
                let end_a = polyline_endpoint_to_chain_end(graph.polylines[a.polyline_index].len(), a.point_index);
                let end_b = polyline_endpoint_to_chain_end(graph.polylines[b.polyline_index].len(), b.point_index);
                if let (Some(ea), Some(eb)) = (end_a, end_b) {
                    let (pid_a, _) = chain_endpoint(seg_a, ea);
                    let (pid_b, _) = chain_endpoint(seg_b, eb);
                    let alpha_a = if !ea { 1.0 } else { 0.0 }; // end_a == "end" -> 1.0
                    let alpha_b = if eb { 0.0 } else { 1.0 }; // end_b == "start" -> 0.0
                    out.g1.push((pid_a, alpha_a, pid_b, alpha_b));
                }
            }
        }
    }
    out
}

fn build_constraints(graph: &StrokeGraph, segments: &[FittedSegment], prims: &[Prim], cfg: &FitConfig) -> SoftConstraints {
    let mut soft = build_junction_constraints(graph, segments, cfg.smooth_junction_deg_threshold, prims);
    let smooth_thresh_rad = cfg.smooth_junction_deg_threshold.to_radians();
    for seg in segments {
        let pids = &seg.primitive_ids;
        for k in 0..pids.len().saturating_sub(1) {
            let a_pid = pids[k];
            let b_pid = pids[k + 1];
            let a_c = prims[a_pid].is_circle();
            let b_c = prims[b_pid].is_circle();
            if a_c && !b_c {
                soft.on_curve.push(((b_pid, true), a_pid));
            } else if b_c && !a_c {
                soft.on_curve.push(((a_pid, false), b_pid));
            } else {
                soft.coincide.push(((a_pid, false), (b_pid, true)));
            }
            if a_c || b_c {
                continue;
            }
            let t_end = tangent_at_end(&prims[a_pid], false);
            let t_start = tangent_at_end(&prims[b_pid], true);
            let d = dot(t_end, t_start).clamp(-1.0, 1.0);
            let deflection = d.lacos();
            if deflection < smooth_thresh_rad {
                soft.g1.push((a_pid, 1.0, b_pid, 0.0));
            }
        }
        if pids.len() >= 2 {
            let poly = &graph.polylines[seg.polyline_index];
            if is_near_closed_polyline(poly, 0.10, 5.0) {
                let first = pids[0];
                let last = *pids.last().unwrap();
                if !(prims[first].is_circle() || prims[last].is_circle()) {
                    soft.coincide.push(((last, false), (first, true)));
                }
            }
        }
    }
    soft
}

pub struct LowGeometryResult {
    pub commands_consolidated: Vec<Command>,
    pub prims_initial: Vec<Prim>,
    pub prims_fitted: Vec<Prim>,
    pub prims_consolidated: Vec<Prim>,
    pub tour: Vec<(usize, bool)>,
}

pub fn vectorize_low_geometry(graph: &StrokeGraph, start_pos: P2, start_heading: f64) -> LowGeometryResult {
    let cfg = FitConfig::default();
    let weights = Weights::default();
    let max_iters = 80usize;
    let mut timer = crate::StageTimer::new();
    let mut segments = build_chains(graph, &cfg);
    timer.lap("  fit chains");
    let prims_initial = assign_global_ids(&mut segments);
    let prims_initial_snapshot = prims_initial.clone();
    let mut source_points: Vec<Vec<P2>> = vec![Vec::new(); prims_initial.len()];
    for seg in &segments {
        for (pid, src) in seg.primitive_ids.iter().zip(&seg.source_points) {
            source_points[*pid] = src.clone();
        }
    }
    let soft_initial = build_constraints(graph, &segments, &prims_initial, &cfg);
    let pos_scale = solve::bbox_diag(&graph.polylines);
    let mut prims_fitted = if !prims_initial.is_empty() {
        solve_once(&prims_initial, &source_points, &soft_initial, weights, pos_scale, max_iters)
    } else {
        vec![]
    };
    timer.lap("  solve 1");
    let mut soft_initial = soft_initial;
    if !prims_fitted.is_empty() {
        let mut new_segments = Vec::new();
        for seg in &segments {
            let chain_prims: Vec<Prim> = seg.primitive_ids.iter().map(|&pid| prims_fitted[pid].clone()).collect();
            let (fused_pieces, fused_src) = fuse_chain(&seg.pieces, &seg.source_points, &chain_prims);
            new_segments.push(FittedSegment {
                polyline_index: seg.polyline_index,
                pieces: fused_pieces,
                primitive_ids: vec![],
                source_points: fused_src,
            });
        }
        segments = new_segments;
        prims_fitted = assign_global_ids(&mut segments);
        source_points = vec![Vec::new(); prims_fitted.len()];
        for seg in &segments {
            for (pid, src) in seg.primitive_ids.iter().zip(&seg.source_points) {
                source_points[*pid] = src.clone();
            }
        }
        soft_initial = build_constraints(graph, &segments, &prims_fitted, &cfg);
    }
    let mut prims_consolidated = if !prims_fitted.is_empty() {
        let additions = beautify::detect(&prims_fitted, &beautify::BeautifyTolerances::default());
        let mut soft_b = soft_initial.clone();
        soft_b.parallel.extend(additions.parallel);
        soft_b.perpendicular.extend(additions.perpendicular);
        soft_b.equal_radius.extend(additions.equal_radius);
        soft_b.concentric.extend(additions.concentric);
        solve_once(&prims_fitted, &source_points, &soft_b, weights, pos_scale, max_iters)
    } else {
        prims_fitted.clone()
    };
    timer.lap("  solve 2 (+fuse, beautify)");
    prims_consolidated = beautify::merge_arc_pairs(&prims_consolidated);
    let snap_tol = 1.5;
    let pen_up_join_tol = 0.5;
    let tour = routing::order_primitives(&prims_consolidated, start_pos, snap_tol);
    let commands = routing::to_commands(&prims_consolidated, &tour, start_pos, start_heading, pen_up_join_tol);
    timer.lap("  routing");
    LowGeometryResult {
        commands_consolidated: commands,
        prims_initial: prims_initial_snapshot,
        prims_fitted,
        prims_consolidated,
        tour,
    }
}
