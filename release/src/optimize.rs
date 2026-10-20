//! Time-optimal command re-ordering (`optimize.py`).

use crate::fmath::DetMath;
use crate::commands::Command;
use crate::config::Config;
use crate::numeric::{norm2, py_mod, sub, P2};

const MAX_TURN_SPEED: f64 = 3200.0;
const ACCELERATION_MAX_DRAWING: f64 = 1000.0;
const WHEELBASE_RADIUS: f64 = 2.93;
const STEP_LENGTH: f64 = 0.055;
const MICROSTEPS_PER_STEP: f64 = 16.0;
const INCHES_PER_MICROSTEP: f64 = STEP_LENGTH / MICROSTEPS_PER_STEP;

fn estimate_motion_time(distance_steps: f64, max_speed: f64, acceleration: f64) -> f64 {
    let d = distance_steps.abs();
    if d <= 0.0 || max_speed <= 0.0 || acceleration <= 0.0 {
        return 0.0;
    }
    let ramp = (max_speed * max_speed) / acceleration;
    if d <= ramp {
        return 2.0 * (d / acceleration).sqrt();
    }
    let accel_time = max_speed / acceleration;
    let cruise = d - ramp;
    2.0 * accel_time + cruise / max_speed
}

fn estimate_line_time(distance_steps: f64) -> f64 {
    estimate_motion_time(distance_steps.abs(), 2000.0, ACCELERATION_MAX_DRAWING)
}

fn estimate_arc_time(radius_inches: f64, angle_deg: f64) -> f64 {
    let angle_deg = angle_deg.abs();
    let outer_radius = WHEELBASE_RADIUS + radius_inches;
    let outer_circ = 2.0 * std::f64::consts::PI * outer_radius;
    let outer_distance = (outer_circ * (angle_deg / 360.0)) / INCHES_PER_MICROSTEP;
    let inner_radius = (radius_inches - WHEELBASE_RADIUS).abs();
    let inner_circ = 2.0 * std::f64::consts::PI * inner_radius;
    let inner_distance = (inner_circ * (angle_deg / 360.0)) / INCHES_PER_MICROSTEP;
    if outer_distance <= 0.0 {
        return 0.0;
    }
    let outer_t = outer_distance / MAX_TURN_SPEED;
    let inner_speed = if outer_t > 0.0 { (inner_distance / outer_t).abs() } else { 0.0 };
    let (outer_accel, inner_accel);
    if inner_speed <= MAX_TURN_SPEED {
        outer_accel = ACCELERATION_MAX_DRAWING;
        inner_accel = (outer_accel * inner_speed) / MAX_TURN_SPEED;
    } else {
        inner_accel = ACCELERATION_MAX_DRAWING;
        outer_accel = (inner_accel * MAX_TURN_SPEED) / inner_speed;
    }
    let outer_time = estimate_motion_time(outer_distance, MAX_TURN_SPEED, outer_accel);
    let inner_time = estimate_motion_time(inner_distance, inner_speed, inner_accel);
    outer_time.max(inner_time)
}

fn estimate_spin_time(angle_deg: f64) -> f64 {
    estimate_arc_time(0.0, angle_deg)
}

pub fn estimate_total_time(cmds: &[Command], ppi: f64) -> f64 {
    let mut total = 0.0;
    for c in cmds {
        match c {
            Command::Spin { degrees } => total += estimate_spin_time(*degrees),
            Command::Line { distance, .. } => {
                let inches = distance / ppi;
                total += estimate_line_time(inches / INCHES_PER_MICROSTEP);
            }
            Command::Arc { radius, degrees } => total += estimate_arc_time(radius / ppi, *degrees),
        }
    }
    total
}

fn wrap_to_pi(a: f64) -> f64 {
    py_mod(a + std::f64::consts::PI, 2.0 * std::f64::consts::PI) - std::f64::consts::PI
}
fn heading_change_deg(from: f64, to: f64) -> f64 {
    wrap_to_pi(to - from).to_degrees()
}
fn heading_of(v: P2) -> f64 {
    v[1].latan2(v[0])
}

fn step_command(pos: P2, heading: f64, cmd: &Command) -> (P2, f64) {
    match cmd {
        Command::Spin { degrees } => (pos, heading + degrees.to_radians()),
        Command::Line { distance, .. } => ([pos[0] + distance * heading.lcos(), pos[1] + distance * heading.lsin()], heading),
        Command::Arc { radius, degrees } => {
            let r = *radius;
            let sweep = degrees.to_radians();
            let ccw = sweep > 0.0;
            let normal_angle = heading + if ccw { std::f64::consts::PI / 2.0 } else { -std::f64::consts::PI / 2.0 };
            let center = [pos[0] + r * normal_angle.lcos(), pos[1] + r * normal_angle.lsin()];
            let start_a = (pos[1] - center[1]).latan2(pos[0] - center[0]);
            let end_a = start_a + sweep;
            let new_pos = [center[0] + r * end_a.lcos(), center[1] + r * end_a.lsin()];
            (new_pos, heading + sweep)
        }
    }
}

struct Primitive {
    cmd: Command,
    entry_pos: P2,
    entry_heading: f64,
    exit_pos: P2,
    exit_heading: f64,
    draw_time: f64,
}

fn extract_primitives(cmds: &[Command], start_pos: P2, start_heading: f64, ppi: f64) -> Vec<Primitive> {
    let mut pos = start_pos;
    let mut heading = start_heading;
    let mut out = Vec::new();
    for cmd in cmds {
        match cmd {
            Command::Spin { degrees } => {
                heading += degrees.to_radians();
                continue;
            }
            Command::Line { pen_down: false, .. } => {
                let (p, h) = step_command(pos, heading, cmd);
                pos = p;
                heading = h;
                continue;
            }
            _ => {}
        }
        let entry_pos = pos;
        let entry_heading = heading;
        let (new_pos, new_heading) = step_command(pos, heading, cmd);
        let draw_time = match cmd {
            Command::Line { distance, .. } => estimate_line_time((distance / ppi) / INCHES_PER_MICROSTEP),
            Command::Arc { radius, degrees } => estimate_arc_time(radius / ppi, *degrees),
            _ => unreachable!(),
        };
        out.push(Primitive { cmd: cmd.clone(), entry_pos, entry_heading, exit_pos: new_pos, exit_heading: new_heading, draw_time });
        pos = new_pos;
        heading = new_heading;
    }
    out
}

struct CostCache {
    n: usize,
    trans: Vec<f64>, // (n*2) x (n*2)
    trans_start: Vec<f64>, // n*2
    draw_sum: f64,
}
impl CostCache {
    #[inline]
    fn t(&self, a: usize, ra: usize, b: usize, rb: usize) -> f64 {
        self.trans[(a * 2 + ra) * (2 * self.n) + (b * 2 + rb)]
    }
    #[inline]
    fn ts(&self, b: usize, rb: usize) -> f64 {
        self.trans_start[b * 2 + rb]
    }
}

fn vec_line_time_with(d: f64, max_speed: f64, accel: f64) -> f64 {
    let d = d.abs();
    let ramp = (max_speed * max_speed) / accel;
    let triangle = 2.0 * (d.max(0.0) / accel).sqrt();
    let accel_time = max_speed / accel;
    let cruise = (d - ramp).max(0.0) / max_speed;
    let trapezoid = 2.0 * accel_time + cruise;
    let out = if d <= ramp { triangle } else { trapezoid };
    if d > 0.0 {
        out
    } else {
        0.0
    }
}
fn vec_line_time(d: f64) -> f64 {
    vec_line_time_with(d, 2000.0, ACCELERATION_MAX_DRAWING)
}
fn vec_spin_time(angle_deg: f64) -> f64 {
    let angle = angle_deg.abs();
    let circ = 2.0 * std::f64::consts::PI * WHEELBASE_RADIUS;
    let distance = (circ * (angle / 360.0)) / INCHES_PER_MICROSTEP;
    vec_line_time_with(distance, MAX_TURN_SPEED, ACCELERATION_MAX_DRAWING)
}

fn short_deg(delta: f64) -> f64 {
    (py_mod(delta + std::f64::consts::PI, 2.0 * std::f64::consts::PI) - std::f64::consts::PI).to_degrees().abs()
}

fn transition(exit_pos: P2, exit_h: f64, entry_pos: P2, entry_h: f64, ppi: f64, tol: f64) -> f64 {
    let gap_vec = sub(entry_pos, exit_pos);
    let gap = crate::numeric::norm2_naive(gap_vec);
    if gap > tol {
        let gap_heading = gap_vec[1].latan2(gap_vec[0]);
        let spin1 = vec_spin_time(short_deg(gap_heading - exit_h));
        let spin2 = vec_spin_time(short_deg(entry_h - gap_heading));
        let line = vec_line_time((gap / ppi) / INCHES_PER_MICROSTEP);
        spin1 + line + spin2
    } else {
        vec_spin_time(short_deg(entry_h - exit_h))
    }
}

fn build_cost_cache(prims: &[Primitive], start_pos: P2, start_heading: f64, ppi: f64, tol: f64) -> CostCache {
    let n = prims.len();
    let mut entry_pos = vec![[0.0; 2]; 2 * n];
    let mut entry_h = vec![0.0; 2 * n];
    let mut exit_pos = vec![[0.0; 2]; 2 * n];
    let mut exit_h = vec![0.0; 2 * n];
    let mut draw_sum = 0.0;
    for (i, p) in prims.iter().enumerate() {
        entry_pos[2 * i] = p.entry_pos;
        entry_pos[2 * i + 1] = p.exit_pos;
        entry_h[2 * i] = p.entry_heading;
        entry_h[2 * i + 1] = p.exit_heading + std::f64::consts::PI;
        exit_pos[2 * i] = p.exit_pos;
        exit_pos[2 * i + 1] = p.entry_pos;
        exit_h[2 * i] = p.exit_heading;
        exit_h[2 * i + 1] = p.entry_heading + std::f64::consts::PI;
        draw_sum += p.draw_time;
    }
    let m = 2 * n;
    let mut trans = vec![0.0; m * m];
    for a in 0..m {
        for b in 0..m {
            trans[a * m + b] = transition(exit_pos[a], exit_h[a], entry_pos[b], entry_h[b], ppi, tol);
        }
    }
    let mut trans_start = vec![0.0; m];
    for b in 0..m {
        trans_start[b] = transition(start_pos, start_heading, entry_pos[b], entry_h[b], ppi, tol);
    }
    CostCache { n, trans, trans_start, draw_sum }
}

#[derive(Clone)]
struct Tour {
    order: Vec<usize>,
    rev: Vec<usize>,
}

fn tour_cost(t: &Tour, c: &CostCache) -> f64 {
    if t.order.is_empty() {
        return 0.0;
    }
    let mut total = c.ts(t.order[0], t.rev[0]);
    for k in 0..t.order.len() - 1 {
        total += c.t(t.order[k], t.rev[k], t.order[k + 1], t.rev[k + 1]);
    }
    total + c.draw_sum
}

fn edge_cost(c: &CostCache, prev: Option<(usize, usize)>, node: (usize, usize), is_first: bool) -> f64 {
    if is_first {
        return c.ts(node.0, node.1);
    }
    let p = prev.unwrap();
    c.t(p.0, p.1, node.0, node.1)
}

fn nearest_neighbor(c: &CostCache, n: usize) -> Tour {
    let mut used = vec![false; n];
    let mut order = Vec::new();
    let mut rev = Vec::new();
    let best = crate::numeric::np_argmin(&c.trans_start);
    let (j0, r0) = (best / 2, best % 2);
    order.push(j0);
    rev.push(r0);
    used[j0] = true;
    let (mut cur, mut cur_r) = (j0, r0);
    for _ in 0..n.saturating_sub(1) {
        let m = 2 * n;
        let row = &c.trans[(cur * 2 + cur_r) * m..(cur * 2 + cur_r + 1) * m];
        let masked: Vec<f64> = (0..m).map(|k| if used[k / 2] { f64::INFINITY } else { row[k] }).collect();
        let idx = crate::numeric::np_argmin(&masked);
        let (j, r) = (idx / 2, idx % 2);
        order.push(j);
        rev.push(r);
        used[j] = true;
        cur = j;
        cur_r = r;
    }
    Tour { order, rev }
}

fn two_opt(tour: Tour, c: &CostCache, max_passes: usize) -> Tour {
    let n = tour.order.len();
    if n < 3 {
        return tour;
    }
    let mut order = tour.order;
    let mut rev = tour.rev;
    for _ in 0..max_passes {
        let mut improved = false;
        for i in 0..n - 1 {
            for j in i + 1..n {
                let (oi, ri) = (order[i], rev[i]);
                let (oj, rj) = (order[j], rev[j]);
                let (old_enter, new_enter) = if i == 0 {
                    (edge_cost(c, None, (oi, ri), true), edge_cost(c, None, (oj, 1 - rj), true))
                } else {
                    let prev = (order[i - 1], rev[i - 1]);
                    (edge_cost(c, Some(prev), (oi, ri), false), edge_cost(c, Some(prev), (oj, 1 - rj), false))
                };
                let (old_leave, new_leave) = if j == n - 1 {
                    (0.0, 0.0)
                } else {
                    let succ = (order[j + 1], rev[j + 1]);
                    (edge_cost(c, Some((oj, rj)), succ, false), edge_cost(c, Some((oi, 1 - ri)), succ, false))
                };
                let delta = (new_enter + new_leave) - (old_enter + old_leave);
                if delta < -1e-9 {
                    order[i..=j].reverse();
                    let seg: Vec<usize> = rev[i..=j].iter().rev().map(|&x| 1 - x).collect();
                    rev[i..=j].copy_from_slice(&seg);
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    Tour { order, rev }
}

fn or_opt(tour: Tour, c: &CostCache, max_segment_len: usize, max_passes: usize) -> Tour {
    let n = tour.order.len();
    if n < 4 {
        return tour;
    }
    let mut order = tour.order;
    let mut rev = tour.rev;
    for _ in 0..max_passes {
        let mut improved = false;
        for seg_len in 1..=max_segment_len {
            let mut i = 0usize;
            while i + seg_len <= n {
                let pre = if i > 0 { Some((order[i - 1], rev[i - 1])) } else { None };
                let has_post = i + seg_len < n;
                let post = if has_post { Some((order[i + seg_len], rev[i + seg_len])) } else { None };
                let seg_nodes: Vec<(usize, usize)> = (0..seg_len).map(|k| (order[i + k], rev[i + k])).collect();
                let rem: Vec<(usize, usize)> =
                    (0..i).map(|k| (order[k], rev[k])).chain((i + seg_len..n).map(|k| (order[k], rev[k]))).collect();
                let m = rem.len();
                let d1 = edge_cost(c, pre, seg_nodes[0], i == 0);
                let d2 = match post {
                    Some(p) => edge_cost(c, Some(seg_nodes[seg_len - 1]), p, false),
                    None => 0.0,
                };
                let g_close = match post {
                    Some(p) => edge_cost(c, pre, p, pre.is_none()),
                    None => 0.0,
                };
                let mut best_delta = -1e-9;
                let mut best_move: Option<(usize, bool)> = None;
                for g in 0..=m {
                    let rg_prev = if g > 0 { Some(rem[g - 1]) } else { None };
                    let rg = if g < m { Some(rem[g]) } else { None };
                    let r_split = match rg {
                        Some(r) => edge_cost(c, rg_prev, r, g == 0),
                        None => 0.0,
                    };
                    for flip in [false, true] {
                        let (seg_first, seg_last) = if flip {
                            ((seg_nodes[seg_len - 1].0, 1 - seg_nodes[seg_len - 1].1), (seg_nodes[0].0, 1 - seg_nodes[0].1))
                        } else {
                            (seg_nodes[0], seg_nodes[seg_len - 1])
                        };
                        let a1 = edge_cost(c, rg_prev, seg_first, g == 0);
                        let a2 = match rg {
                            Some(r) => edge_cost(c, Some(seg_last), r, false),
                            None => 0.0,
                        };
                        let delta = (g_close + a1 + a2) - (d1 + d2 + r_split);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some((g, flip));
                        }
                    }
                }
                if let Some((g, flip)) = best_move {
                    let seg_out: Vec<(usize, usize)> = if flip {
                        seg_nodes.iter().rev().map(|&(p, r)| (p, 1 - r)).collect()
                    } else {
                        seg_nodes.clone()
                    };
                    let mut new_nodes: Vec<(usize, usize)> = rem[..g].to_vec();
                    new_nodes.extend(seg_out);
                    new_nodes.extend_from_slice(&rem[g..]);
                    order = new_nodes.iter().map(|x| x.0).collect();
                    rev = new_nodes.iter().map(|x| x.1).collect();
                    improved = true;
                    i = 0;
                    continue;
                }
                i += 1;
            }
        }
        if !improved {
            break;
        }
    }
    Tour { order, rev }
}

fn emit_transition(cur_pos: P2, cur_heading: f64, entry_pos: P2, entry_heading: f64, tol: f64, out: &mut Vec<Command>) -> (P2, f64) {
    let mut cur_pos = cur_pos;
    let mut cur_heading = cur_heading;
    let gap_vec = sub(entry_pos, cur_pos);
    let gap = norm2(gap_vec);
    if gap > tol {
        let gh = heading_of(gap_vec);
        let s1 = heading_change_deg(cur_heading, gh);
        if s1.abs() > 1e-3 {
            out.push(Command::Spin { degrees: s1 });
            cur_heading = gh;
        }
        out.push(Command::Line { distance: gap, pen_down: false });
        cur_pos = entry_pos;
    }
    let s2 = heading_change_deg(cur_heading, entry_heading);
    if s2.abs() > 1e-3 {
        out.push(Command::Spin { degrees: s2 });
        cur_heading = entry_heading;
    }
    (cur_pos, cur_heading)
}

fn emit_tour(t: &Tour, prims: &[Primitive], start_pos: P2, start_heading: f64, tol: f64) -> Vec<Command> {
    let mut out = Vec::new();
    let mut cur_pos = start_pos;
    let mut cur_heading = start_heading;
    for (&idx, &rev) in t.order.iter().zip(&t.rev) {
        let p = &prims[idx];
        let (ep, eh) = if rev == 0 { (p.entry_pos, p.entry_heading) } else { (p.exit_pos, p.exit_heading + std::f64::consts::PI) };
        let (np_, nh) = emit_transition(cur_pos, cur_heading, ep, eh, tol, &mut out);
        cur_pos = np_;
        cur_heading = nh;
        let cmd = if rev == 1 {
            match &p.cmd {
                Command::Line { distance, pen_down } => Command::Line { distance: *distance, pen_down: *pen_down },
                Command::Arc { radius, degrees } => Command::Arc { radius: *radius, degrees: -degrees },
                _ => unreachable!(),
            }
        } else {
            p.cmd.clone()
        };
        out.push(cmd.clone());
        let (np2, nh2) = step_command(cur_pos, cur_heading, &cmd);
        cur_pos = np2;
        cur_heading = nh2;
    }
    out
}

pub fn optimize_route(commands: &[Command], start_pos: P2, start_heading: f64, cfg: &Config) -> Vec<Command> {
    let ppi = cfg.opt_pixels_per_inch;
    let tol = cfg.opt_pen_up_join_tol;
    let before = estimate_total_time(commands, ppi);
    let prims = extract_primitives(commands, start_pos, start_heading, ppi);
    if prims.is_empty() {
        return commands.to_vec();
    }
    let n = prims.len();
    let cache = build_cost_cache(&prims, start_pos, start_heading, ppi, tol);
    let seeds = vec![nearest_neighbor(&cache, n), Tour { order: (0..n).collect(), rev: vec![0; n] }];
    let mut best_tour: Option<Tour> = None;
    let mut best_cost = f64::INFINITY;
    for seed in seeds {
        let t = two_opt(seed, &cache, cfg.opt_two_opt_passes);
        let t = or_opt(t, &cache, cfg.opt_or_opt_max_segment_len, cfg.opt_or_opt_passes);
        let c = tour_cost(&t, &cache);
        if c < best_cost {
            best_cost = c;
            best_tour = Some(t);
        }
    }
    let best = best_tour.unwrap();
    let out = emit_tour(&best, &prims, start_pos, start_heading, tol);
    let after = estimate_total_time(&out, ppi);
    if after > before + 1e-9 {
        commands.to_vec()
    } else {
        out
    }
}
