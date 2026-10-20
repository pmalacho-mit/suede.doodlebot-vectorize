//! Geometric primitives (`primitives.py`).

use crate::fmath::DetMath;
use crate::numeric::{dist, norm2, sub, P2};

pub const EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
pub enum Prim {
    Line { p0: P2, p1: P2 },
    Arc { p0: P2, p1: P2, bulge: f64 },
    Circle { center: P2, radius: f64 },
}

impl Prim {
    pub fn is_line(&self) -> bool {
        matches!(self, Prim::Line { .. })
    }
    pub fn is_arc(&self) -> bool {
        matches!(self, Prim::Arc { .. })
    }
    pub fn is_circle(&self) -> bool {
        matches!(self, Prim::Circle { .. })
    }
    pub fn p0(&self) -> P2 {
        match self {
            Prim::Line { p0, .. } | Prim::Arc { p0, .. } => *p0,
            Prim::Circle { center, radius } => [center[0] + radius, center[1]],
        }
    }
    pub fn p1(&self) -> P2 {
        match self {
            Prim::Line { p1, .. } | Prim::Arc { p1, .. } => *p1,
            Prim::Circle { center, radius } => [center[0] + radius, center[1]],
        }
    }
    pub fn param_count(&self) -> usize {
        match self {
            Prim::Line { .. } => 4,
            Prim::Arc { .. } => 5,
            Prim::Circle { .. } => 3,
        }
    }
}

// Line helpers
pub fn line_length(p0: P2, p1: P2) -> f64 {
    dist(p1, p0)
}
pub fn line_direction(p0: P2, p1: P2) -> P2 {
    let d = sub(p1, p0);
    let n = norm2(d);
    if n < EPS {
        [1.0, 0.0]
    } else {
        [d[0] / n, d[1] / n]
    }
}
pub fn line_normal(p0: P2, p1: P2) -> P2 {
    let d = line_direction(p0, p1);
    [-d[1], d[0]]
}
/// `Line.perpendicular_distance(pts)` for one of `k` rows (`(pts - p0) @ n`, a gemv).
pub fn line_perp_distance_k(p0: P2, p1: P2, pt: P2, k: usize) -> f64 {
    let n = line_normal(p0, p1);
    crate::blas::gemv_row([pt[0] - p0[0], pt[1] - p0[1]], n, k)
}

// Arc helpers
pub fn arc_chord(p0: P2, p1: P2) -> f64 {
    dist(p1, p0)
}
pub fn arc_sweep(bulge: f64) -> f64 {
    4.0 * bulge.latan()
}
pub fn arc_radius(p0: P2, p1: P2, bulge: f64) -> f64 {
    let l = arc_chord(p0, p1);
    if bulge.abs() < EPS {
        return f64::INFINITY;
    }
    l * (1.0 + bulge * bulge) / (4.0 * bulge.abs())
}
pub fn arc_center(p0: P2, p1: P2, bulge: f64) -> P2 {
    let l = arc_chord(p0, p1);
    if l < EPS || bulge.abs() < EPS {
        return [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
    }
    let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
    let cd = [(p1[0] - p0[0]) / l, (p1[1] - p0[1]) / l];
    let normal = [-cd[1], cd[0]];
    let h = l * (1.0 - bulge * bulge) / (4.0 * bulge);
    [mid[0] + h * normal[0], mid[1] + h * normal[1]]
}
pub fn arc_theta0(p0: P2, p1: P2, bulge: f64) -> f64 {
    let c = arc_center(p0, p1, bulge);
    (p0[1] - c[1]).latan2(p0[0] - c[0])
}
pub fn arc_tangent_at(p0: P2, p1: P2, bulge: f64, alpha: f64) -> P2 {
    let sweep = arc_sweep(bulge);
    if sweep.abs() < EPS {
        let d = sub(p1, p0);
        let n = norm2(d);
        return if n > EPS { [d[0] / n, d[1] / n] } else { [1.0, 0.0] };
    }
    let theta = arc_theta0(p0, p1, bulge) + alpha * sweep;
    let sign = if sweep > 0.0 { 1.0 } else { -1.0 };
    [sign * -theta.lsin(), sign * theta.lcos()]
}
/// `Arc.signed_distance(pts)` for one of `k` rows.
pub fn arc_signed_distance_k(p0: P2, p1: P2, bulge: f64, pt: P2, k: usize) -> f64 {
    let c = arc_center(p0, p1, bulge);
    let r = arc_radius(p0, p1, bulge);
    if !r.is_finite() {
        // line perpendicular distance fallback
        let d = sub(p1, p0);
        let n = norm2(d);
        if n < EPS {
            return crate::numeric::dist_naive(pt, p0);
        }
        let u = [d[0] / n, d[1] / n];
        let perp = [-u[1], u[0]];
        return crate::blas::gemv_row([pt[0] - p0[0], pt[1] - p0[1]], perp, k);
    }
    crate::numeric::dist_naive(pt, c) - r
}

// Circle helpers
pub fn circle_tangent_at(alpha: f64) -> P2 {
    let theta = 2.0 * std::f64::consts::PI * alpha;
    [-theta.lsin(), theta.lcos()]
}

pub fn tangent_at_end(p: &Prim, end_is_start: bool) -> P2 {
    match p {
        Prim::Line { p0, p1 } => line_direction(*p0, *p1),
        Prim::Arc { p0, p1, bulge } => arc_tangent_at(*p0, *p1, *bulge, if end_is_start { 0.0 } else { 1.0 }),
        Prim::Circle { .. } => circle_tangent_at(0.0),
    }
}

pub fn prim_radius(p: &Prim) -> f64 {
    match p {
        Prim::Circle { radius, .. } => *radius,
        Prim::Arc { p0, p1, bulge } => arc_radius(*p0, *p1, *bulge),
        Prim::Line { .. } => f64::INFINITY,
    }
}
pub fn prim_center(p: &Prim) -> Option<P2> {
    match p {
        Prim::Circle { center, .. } => Some(*center),
        Prim::Arc { p0, p1, bulge } => Some(arc_center(*p0, *p1, *bulge)),
        Prim::Line { .. } => None,
    }
}
