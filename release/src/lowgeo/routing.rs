//! Eulerian routing and command emission (`routing.py`).

use crate::fmath::DetMath;
use super::nxgraph::{Key, MultiGraph};
use super::primitives::*;
use crate::commands::Command;
use crate::numeric::{dist, py_mod, sub, P2};

fn endpoint_locations(prims: &[Prim], snap_tol: f64) -> (Vec<(usize, usize)>, Vec<P2>) {
    let mut pts: Vec<P2> = Vec::new();
    let mut refs: Vec<(usize, bool)> = Vec::new();
    for (pid, p) in prims.iter().enumerate() {
        if p.is_circle() {
            let v = p.p0();
            pts.push(v);
            refs.push((pid, true));
            pts.push(v);
            refs.push((pid, false));
        } else {
            pts.push(p.p0());
            refs.push((pid, true));
            pts.push(p.p1());
            refs.push((pid, false));
        }
    }
    let mut cluster = vec![usize::MAX; pts.len()];
    let mut centroids: Vec<Vec<P2>> = Vec::new();
    for (i, &p) in pts.iter().enumerate() {
        let mut best: Option<usize> = None;
        let mut best_d = snap_tol;
        for (ci, ps) in centroids.iter().enumerate() {
            let mean = crate::numeric::mean_axis0(ps);
            let d = dist(p, mean);
            if d < best_d {
                best = Some(ci);
                best_d = d;
            }
        }
        match best {
            Some(ci) => {
                cluster[i] = ci;
                centroids[ci].push(p);
            }
            None => {
                cluster[i] = centroids.len();
                centroids.push(vec![p]);
            }
        }
    }
    let vert_positions: Vec<P2> = centroids.iter().map(|ps| crate::numeric::mean_axis0(ps)).collect();
    let mut edge_to_verts = vec![(usize::MAX, usize::MAX); prims.len()];
    for (i, &(pid, is_start)) in refs.iter().enumerate() {
        if is_start {
            edge_to_verts[pid].0 = cluster[i];
        } else {
            edge_to_verts[pid].1 = cluster[i];
        }
    }
    (edge_to_verts, vert_positions)
}

fn build_multigraph(edge_to_verts: &[(usize, usize)], prims: &[Prim]) -> MultiGraph {
    let mut g = MultiGraph::new();
    for (pid, &(u, v)) in edge_to_verts.iter().enumerate() {
        g.add_node(u);
        g.add_node(v);
        let w = match &prims[pid] {
            Prim::Line { p0, p1 } => line_length(*p0, *p1),
            Prim::Arc { p0, p1, bulge } => {
                let r = arc_radius(*p0, *p1, *bulge);
                arc_sweep(*bulge).abs() * (if r.is_finite() { r } else { arc_chord(*p0, *p1) })
            }
            Prim::Circle { radius, .. } => 2.0 * std::f64::consts::PI * radius,
        };
        g.add_edge(u, v, Some(Key::P(pid)), w);
    }
    g
}

fn pick_start_vertex(g: &MultiGraph, vert_positions: &[P2], start_pos: P2) -> usize {
    let nodes = g.nodes();
    let odd: Vec<usize> = nodes.iter().cloned().filter(|&v| g.degree(v) % 2 == 1).collect();
    let candidates = if odd.is_empty() { nodes } else { odd };
    if candidates.is_empty() {
        return 0;
    }
    let mut best = candidates[0];
    let mut best_d = dist(vert_positions[best], start_pos);
    for &v in &candidates[1..] {
        let d = dist(vert_positions[v], start_pos);
        if d < best_d {
            best = v;
            best_d = d;
        }
    }
    best
}

fn eulerize_per_component(g: &MultiGraph) -> MultiGraph {
    let mut out = MultiGraph::new();
    for comp in g.connected_components() {
        let mut sub = g.subgraph_copy(&comp);
        if sub.number_of_edges() == 0 {
            for n in sub.nodes() {
                out.add_node(n);
            }
            continue;
        }
        if !sub.is_eulerian() && !sub.has_eulerian_path(None) {
            sub = sub.eulerize();
        }
        out = MultiGraph::compose(&out, &sub);
    }
    out
}

/// Returns a tour [(prim_id, reverse)].
pub fn order_primitives(prims: &[Prim], start_pos: P2, snap_tol: f64) -> Vec<(usize, bool)> {
    if prims.is_empty() {
        return vec![];
    }
    let (edge_to_verts, vert_positions) = endpoint_locations(prims, snap_tol);
    let g = build_multigraph(&edge_to_verts, prims);
    let g2 = eulerize_per_component(&g);
    let mut tour: Vec<(usize, usize, Key)> = Vec::new();
    let mut cur_pos = start_pos;
    let components = g2.connected_components();
    let mut used = vec![false; components.len()];
    loop {
        let mut best_ci: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        let mut best_start = 0usize;
        for (ci, comp) in components.iter().enumerate() {
            if used[ci] {
                continue;
            }
            let sub = g2.subgraph_copy(comp);
            if sub.number_of_edges() == 0 {
                continue;
            }
            let start_v = pick_start_vertex(&sub, &vert_positions, cur_pos);
            let d = dist(vert_positions[start_v], cur_pos);
            if d < best_d {
                best_d = d;
                best_ci = Some(ci);
                best_start = start_v;
            }
        }
        let ci = match best_ci {
            None => break,
            Some(c) => c,
        };
        let sub = g2.subgraph_copy(&components[ci]);
        used[ci] = true;
        let path = if sub.is_eulerian() { sub.eulerian_circuit(best_start) } else { sub.eulerian_path(best_start) };
        for (u, v, k) in path {
            tour.push((u, v, k));
            cur_pos = vert_positions[v];
        }
    }
    let mut out = Vec::new();
    for (u, v, key) in tour {
        if let Key::P(pid) = key {
            let (ou, ov) = edge_to_verts[pid];
            let reverse = u == ov && v == ou && ou != ov;
            out.push((pid, reverse));
        }
    }
    out
}

fn wrap_to_pi(angle: f64) -> f64 {
    py_mod(angle + std::f64::consts::PI, 2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

fn heading_change_deg(from: f64, to: f64) -> f64 {
    wrap_to_pi(to - from).to_degrees()
}

fn heading_of(v: P2) -> f64 {
    v[1].latan2(v[0])
}

fn reverse_primitive(p: &Prim) -> Prim {
    match p {
        Prim::Line { p0, p1 } => Prim::Line { p0: *p1, p1: *p0 },
        Prim::Arc { p0, p1, bulge } => Prim::Arc { p0: *p1, p1: *p0, bulge: -bulge },
        Prim::Circle { .. } => p.clone(),
    }
}

pub fn to_commands(prims: &[Prim], tour: &[(usize, bool)], start_pos: P2, start_heading: f64, pen_up_join_tol: f64) -> Vec<Command> {
    let mut cmds = Vec::new();
    let mut cur_pos = start_pos;
    let mut cur_heading = start_heading;
    for &(pid, reverse) in tour {
        let prim = if reverse { reverse_primitive(&prims[pid]) } else { prims[pid].clone() };
        let start = prim.p0();
        let gap = dist(start, cur_pos);
        if gap > pen_up_join_tol {
            let target_heading = heading_of(sub(start, cur_pos));
            let spin_deg = heading_change_deg(cur_heading, target_heading);
            if spin_deg.abs() > 1e-3 {
                cmds.push(Command::Spin { degrees: spin_deg });
                cur_heading = target_heading;
            }
            cmds.push(Command::Line { distance: gap, pen_down: false });
            cur_pos = start;
        }
        let mut emit_as_line: Option<f64> = None;
        let mut arc_r = 0.0;
        let mut arc_sweep_deg = 0.0;
        if let Prim::Arc { p0, p1, bulge } = &prim {
            let sweep_rad = arc_sweep(*bulge);
            arc_sweep_deg = sweep_rad.to_degrees();
            let r = arc_radius(*p0, *p1, *bulge);
            arc_r = r;
            let (sagitta, chord_over_r) = if r.is_finite() && r > 0.0 {
                ((r * (1.0 - (sweep_rad / 2.0).lcos())).abs(), arc_chord(*p0, *p1) / r)
            } else {
                (0.0, 0.0)
            };
            let degenerate = !r.is_finite() || r > 1e6 || sagitta < 0.5 || (arc_sweep_deg.abs() > 270.0 && chord_over_r < 0.15);
            if degenerate {
                emit_as_line = Some(arc_chord(*p0, *p1));
            }
        }
        let primitive_start_heading = match &prim {
            Prim::Line { p0, p1 } => heading_of(sub(*p1, *p0)),
            Prim::Arc { p0, p1, .. } if emit_as_line.is_some() => heading_of(sub(*p1, *p0)),
            Prim::Arc { .. } => heading_of(tangent_at_end(&prim, true)),
            Prim::Circle { .. } => heading_of(circle_tangent_at(0.0)),
        };
        let spin_deg = heading_change_deg(cur_heading, primitive_start_heading);
        if spin_deg.abs() > 1e-3 {
            cmds.push(Command::Spin { degrees: spin_deg });
            cur_heading = primitive_start_heading;
        }
        match &prim {
            Prim::Line { p0, p1 } => {
                let d = line_length(*p0, *p1);
                if d > 1e-9 {
                    cmds.push(Command::Line { distance: d, pen_down: true });
                }
                cur_pos = *p1;
                cur_heading = primitive_start_heading;
            }
            Prim::Arc { p1, .. } => {
                if let Some(dl) = emit_as_line {
                    if dl > 1e-9 {
                        cmds.push(Command::Line { distance: dl, pen_down: true });
                    }
                    cur_pos = *p1;
                    cur_heading = primitive_start_heading;
                } else {
                    cmds.push(Command::Arc { radius: arc_r, degrees: arc_sweep_deg });
                    cur_pos = *p1;
                    cur_heading = heading_of(tangent_at_end(&prim, false));
                }
            }
            Prim::Circle { radius, .. } => {
                cmds.push(Command::Arc { radius: *radius, degrees: 360.0 });
            }
        }
    }
    cmds
}
