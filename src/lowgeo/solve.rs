//! Joint least-squares solve: parameter manifest, residual assembly, sparse
//! finite-difference Jacobian and a faithful port of scipy's Trust Region
//! Reflective solver (unbounded, `tr_solver='lsmr'`, 2-D subspace).
//!
//! The residual vector is described as a list of row recipes; every row's
//! arithmetic is independent of the others, so the finite-difference Jacobian
//! only evaluates the rows a column group actually touches (identical values
//! to a full evaluation, a fraction of the work).

use crate::fmath::DetMath;
use super::primitives::*;
use crate::blas::{ddot, ddot_self, dot2, gemv_row, norm as bnorm};
use crate::numeric::{dist, dist_naive, norm2, real_roots, sub, P2};

pub type EndRef = (usize, bool); // (prim id, is_start)

#[derive(Clone, Default, Debug)]
pub struct SoftConstraints {
    pub coincide: Vec<(EndRef, EndRef)>,
    pub on_curve: Vec<(EndRef, usize)>,
    pub g1: Vec<(usize, f64, usize, f64)>,
    pub parallel: Vec<(usize, usize)>,
    pub perpendicular: Vec<(usize, usize)>,
    pub equal_radius: Vec<(usize, usize)>,
    pub concentric: Vec<(usize, usize)>,
}

#[derive(Clone, Copy)]
pub struct Weights {
    pub data: f64,
    pub coincide: f64,
    pub on_curve: f64,
    pub g1: f64,
    pub parallel: f64,
    pub perpendicular: f64,
    pub equal_radius: f64,
    pub concentric: f64,
    pub radius_reg: f64,
    pub endpoint_anchor: f64,
    pub bulge_reg: f64,
}
impl Default for Weights {
    fn default() -> Self {
        Weights {
            data: 1.0,
            coincide: 200.0,
            on_curve: 200.0,
            g1: 10.0,
            parallel: 5.0,
            perpendicular: 5.0,
            equal_radius: 2.0,
            concentric: 5.0,
            radius_reg: 0.05,
            endpoint_anchor: 5.0,
            bulge_reg: 0.5,
        }
    }
}

pub fn pack(prims: &[Prim]) -> Vec<f64> {
    let mut out = Vec::new();
    for p in prims {
        match p {
            Prim::Line { p0, p1 } => out.extend_from_slice(&[p0[0], p0[1], p1[0], p1[1]]),
            Prim::Arc { p0, p1, bulge } => out.extend_from_slice(&[p0[0], p0[1], p1[0], p1[1], *bulge]),
            Prim::Circle { center, radius } => out.extend_from_slice(&[center[0], center[1], *radius]),
        }
    }
    out
}

pub fn unpack(x: &[f64], template: &[Prim]) -> Vec<Prim> {
    let mut out = Vec::with_capacity(template.len());
    let mut c = 0;
    for p in template {
        match p {
            Prim::Line { .. } => {
                out.push(Prim::Line { p0: [x[c], x[c + 1]], p1: [x[c + 2], x[c + 3]] });
                c += 4;
            }
            Prim::Arc { .. } => {
                out.push(Prim::Arc { p0: [x[c], x[c + 1]], p1: [x[c + 2], x[c + 3]], bulge: x[c + 4] });
                c += 5;
            }
            Prim::Circle { .. } => {
                out.push(Prim::Circle { center: [x[c], x[c + 1]], radius: x[c + 2] });
                c += 3;
            }
        }
    }
    out
}

fn parameter_scales(prims: &[Prim], pos_scale: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for p in prims {
        match p {
            Prim::Line { .. } => out.extend_from_slice(&[pos_scale; 4]),
            Prim::Arc { .. } => {
                out.extend_from_slice(&[pos_scale; 4]);
                out.push(1.0);
            }
            Prim::Circle { .. } => out.extend_from_slice(&[pos_scale; 3]),
        }
    }
    out
}

fn param_offsets(prims: &[Prim]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(prims.len());
    let mut c = 0;
    for p in prims {
        let n = p.param_count();
        out.push((c, c + n));
        c += n;
    }
    out
}

/// Per-primitive geometry derived from the parameter vector.
#[derive(Clone, Copy)]
enum PC {
    Line { p0: P2, p1: P2, dir: P2 },
    Arc { p0: P2, p1: P2, bulge: f64, center: P2, radius: f64 },
    Circle { center: P2, radius: f64 },
}

impl PC {
    #[inline]
    fn endpoint(&self, is_start: bool) -> P2 {
        match self {
            PC::Line { p0, p1, .. } | PC::Arc { p0, p1, .. } => {
                if is_start {
                    *p0
                } else {
                    *p1
                }
            }
            PC::Circle { center, radius } => [center[0] + radius, center[1]],
        }
    }
    #[inline]
    fn direction(&self) -> P2 {
        match self {
            PC::Line { dir, .. } => *dir,
            _ => [1.0, 0.0],
        }
    }
    fn tangent_at(&self, alpha: f64) -> P2 {
        match self {
            PC::Line { dir, .. } => *dir,
            PC::Arc { p0, p1, bulge, .. } => {
                if bulge.abs() < 1e-3 {
                    let d = sub(*p1, *p0);
                    let n = norm2(d);
                    if n > EPS {
                        return [d[0] / n, d[1] / n];
                    }
                    return [1.0, 0.0];
                }
                arc_tangent_at(*p0, *p1, *bulge, alpha)
            }
            PC::Circle { .. } => circle_tangent_at(alpha),
        }
    }
    #[inline]
    fn radius(&self) -> f64 {
        match self {
            PC::Circle { radius, .. } => *radius,
            PC::Arc { radius, .. } => *radius,
            PC::Line { .. } => f64::INFINITY,
        }
    }
    #[inline]
    fn center(&self) -> P2 {
        match self {
            PC::Circle { center, .. } => *center,
            PC::Arc { center, .. } => *center,
            PC::Line { p0, .. } => *p0,
        }
    }
}

#[derive(Clone, Copy)]
enum Row {
    Data { pid: usize, pt: usize, k: usize },
    Anchor { pid: usize, which: u8 },
    Coincide { a: EndRef, b: EndRef, comp: u8 },
    OnCurve { t: EndRef, host: usize },
    G1 { a: usize, alpha_a: f64, b: usize, alpha_b: f64 },
    Parallel { a: usize, b: usize },
    Perp { a: usize, b: usize },
    EqualRadius { a: usize, b: usize },
    Concentric { a: usize, b: usize, comp: u8 },
    RadiusReg { pid: usize, r0: f64 },
    Bulge { pid: usize },
}

pub struct Problem<'a> {
    pub template: &'a [Prim],
    pub source_points: &'a [Vec<P2>],
    pub soft: &'a SoftConstraints,
    pub weights: Weights,
    pub initial_radii: Vec<(usize, f64)>,
    rows: Vec<Row>,
    row_cols: Vec<Vec<(usize, usize)>>, // sparsity: parameter slices per row
    slices: Vec<(usize, usize)>,
}

impl<'a> Problem<'a> {
    pub fn new(
        template: &'a [Prim],
        source_points: &'a [Vec<P2>],
        soft: &'a SoftConstraints,
        weights: Weights,
        initial_radii: Vec<(usize, f64)>,
    ) -> Self {
        let slices = param_offsets(template);
        let mut rows = Vec::new();
        let mut row_cols: Vec<Vec<(usize, usize)>> = Vec::new();
        let two = |a: usize, b: usize| -> Vec<(usize, usize)> {
            if a == b {
                vec![slices[a]]
            } else {
                vec![slices[a], slices[b]]
            }
        };
        // 1. data
        for (pid, pts) in source_points.iter().enumerate() {
            if pts.is_empty() {
                continue;
            }
            let k = pts.len();
            for pt in 0..k {
                rows.push(Row::Data { pid, pt, k });
                row_cols.push(vec![slices[pid]]);
            }
        }
        // 1b. anchors
        for (pid, pts) in source_points.iter().enumerate() {
            if pts.is_empty() {
                continue;
            }
            if !template[pid].is_circle() {
                for which in 0..4u8 {
                    rows.push(Row::Anchor { pid, which });
                    row_cols.push(vec![slices[pid]]);
                }
            }
        }
        for &(a, b) in &soft.coincide {
            for comp in 0..2u8 {
                rows.push(Row::Coincide { a, b, comp });
                row_cols.push(two(a.0, b.0));
            }
        }
        for &(t, host) in &soft.on_curve {
            rows.push(Row::OnCurve { t, host });
            row_cols.push(two(t.0, host));
        }
        for &(a, alpha_a, b, alpha_b) in &soft.g1 {
            rows.push(Row::G1 { a, alpha_a, b, alpha_b });
            row_cols.push(two(a, b));
        }
        for &(a, b) in &soft.parallel {
            if template[a].is_line() && template[b].is_line() {
                rows.push(Row::Parallel { a, b });
                row_cols.push(two(a, b));
            }
        }
        for &(a, b) in &soft.perpendicular {
            if template[a].is_line() && template[b].is_line() {
                rows.push(Row::Perp { a, b });
                row_cols.push(two(a, b));
            }
        }
        for &(a, b) in &soft.equal_radius {
            rows.push(Row::EqualRadius { a, b });
            row_cols.push(two(a, b));
        }
        for &(a, b) in &soft.concentric {
            for comp in 0..2u8 {
                rows.push(Row::Concentric { a, b, comp });
                row_cols.push(two(a, b));
            }
        }
        for &(pid, r0) in &initial_radii {
            if r0 <= EPS {
                continue;
            }
            rows.push(Row::RadiusReg { pid, r0 });
            row_cols.push(vec![slices[pid]]);
        }
        for (pid, p) in template.iter().enumerate() {
            if p.is_arc() {
                rows.push(Row::Bulge { pid });
                row_cols.push(vec![slices[pid]]);
            }
        }
        Problem { template, source_points, soft, weights, initial_radii, rows, row_cols, slices }
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn cache(&self, x: &[f64], out: &mut Vec<PC>) {
        out.clear();
        for (i, p) in self.template.iter().enumerate() {
            let c = self.slices[i].0;
            out.push(match p {
                Prim::Line { .. } => {
                    let p0 = [x[c], x[c + 1]];
                    let p1 = [x[c + 2], x[c + 3]];
                    PC::Line { p0, p1, dir: line_direction(p0, p1) }
                }
                Prim::Arc { .. } => {
                    let p0 = [x[c], x[c + 1]];
                    let p1 = [x[c + 2], x[c + 3]];
                    let bulge = x[c + 4];
                    PC::Arc { p0, p1, bulge, center: arc_center(p0, p1, bulge), radius: arc_radius(p0, p1, bulge) }
                }
                Prim::Circle { .. } => PC::Circle { center: [x[c], x[c + 1]], radius: x[c + 2] },
            });
        }
    }

    #[inline]
    fn row_value(&self, pc: &[PC], row: &Row) -> f64 {
        let w = &self.weights;
        match *row {
            Row::Data { pid, pt, k } => {
                let p = self.source_points[pid][pt];
                match pc[pid] {
                    PC::Line { p0, dir, .. } => {
                        let n = [-dir[1], dir[0]];
                        w.data * gemv_row([p[0] - p0[0], p[1] - p0[1]], n, k)
                    }
                    PC::Arc { p0, p1, center, radius, .. } => {
                        if !radius.is_finite() {
                            // `_line_perp` fallback
                            let d = sub(p1, p0);
                            let n = norm2(d);
                            if n < EPS {
                                return w.data * dist_naive(p, p0);
                            }
                            let u = [d[0] / n, d[1] / n];
                            let perp = [-u[1], u[0]];
                            return w.data * gemv_row([p[0] - p0[0], p[1] - p0[1]], perp, k);
                        }
                        w.data * (dist_naive(p, center) - radius)
                    }
                    PC::Circle { center, radius } => w.data * (dist_naive(p, center) - radius),
                }
            }
            Row::Anchor { pid, which } => {
                let pts = &self.source_points[pid];
                let (p0, p1) = match pc[pid] {
                    PC::Line { p0, p1, .. } | PC::Arc { p0, p1, .. } => (p0, p1),
                    _ => unreachable!(),
                };
                let a = pts[0];
                let b = pts[pts.len() - 1];
                match which {
                    0 => w.endpoint_anchor * (p0[0] - a[0]),
                    1 => w.endpoint_anchor * (p0[1] - a[1]),
                    2 => w.endpoint_anchor * (p1[0] - b[0]),
                    _ => w.endpoint_anchor * (p1[1] - b[1]),
                }
            }
            Row::Coincide { a, b, comp } => {
                let pa = pc[a.0].endpoint(a.1);
                let pb = pc[b.0].endpoint(b.1);
                w.coincide * (pa[comp as usize] - pb[comp as usize])
            }
            Row::OnCurve { t, host } => {
                let p = pc[t.0].endpoint(t.1);
                let v = match pc[host] {
                    PC::Line { p0, dir, .. } => {
                        let n = [-dir[1], dir[0]];
                        gemv_row([p[0] - p0[0], p[1] - p0[1]], n, 1)
                    }
                    PC::Arc { center, radius, .. } => dist(p, center) - radius,
                    PC::Circle { center, radius } => dist(p, center) - radius,
                };
                w.on_curve * v
            }
            Row::G1 { a, alpha_a, b, alpha_b } => {
                let ta = pc[a].tangent_at(alpha_a);
                let tb = pc[b].tangent_at(alpha_b);
                w.g1 * (ta[0] * tb[1] - ta[1] * tb[0])
            }
            Row::Parallel { a, b } => {
                let da = pc[a].direction();
                let db = pc[b].direction();
                w.parallel * (da[0] * db[1] - da[1] * db[0])
            }
            Row::Perp { a, b } => {
                let da = pc[a].direction();
                let db = pc[b].direction();
                w.perpendicular * dot2(da, db)
            }
            Row::EqualRadius { a, b } => w.equal_radius * (pc[a].radius() - pc[b].radius()),
            Row::Concentric { a, b, comp } => {
                let ca = pc[a].center();
                let cb = pc[b].center();
                w.concentric * (ca[comp as usize] - cb[comp as usize])
            }
            Row::RadiusReg { pid, r0 } => {
                let r = pc[pid].radius();
                w.radius_reg * (r / r0).lln()
            }
            Row::Bulge { pid } => match pc[pid] {
                PC::Arc { bulge, .. } => w.bulge_reg * bulge * bulge,
                _ => unreachable!(),
            },
        }
    }

    /// Full residual vector.
    pub fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let mut pc = Vec::with_capacity(self.template.len());
        self.cache(x, &mut pc);
        self.rows.iter().map(|r| self.row_value(&pc, r)).collect()
    }

    fn residuals_rows(&self, x: &[f64], which: &[usize], pc: &mut Vec<PC>, out: &mut [f64]) {
        self.cache(x, pc);
        for (k, &r) in which.iter().enumerate() {
            out[k] = self.row_value(pc, &self.rows[r]);
        }
    }
}

/// Compressed sparse row Jacobian with canonical (sorted) column indices.
/// A compressed-row pattern re-laid out so that up to `LANES` rows of equal
/// length are processed in lockstep: `vals`/`idx` hold entry k of lane r at
/// `(start + k) * LANES + r`. Each row's dot product is still accumulated in
/// exactly scipy's order (sequentially from 0.0), so results are bit-identical
/// to `csr_matvec`; the lanes only remove the per-row dependency chain.
const LANES: usize = 8;

struct BlockedPattern {
    /// (start slot, row length, lanes used, row ids)
    blocks: Vec<(usize, usize, usize, [usize; LANES])>,
    idx: Vec<usize>,
    /// position in the source (CSR) data array for every slot; usize::MAX = padding
    pos: Vec<usize>,
    nrows: usize,
    /// every index in `idx` is < ncols (checked at construction)
    ncols: usize,
}

impl BlockedPattern {
    fn new(ptr: &[usize], indices: &[usize]) -> Self {
        let nrows = ptr.len() - 1;
        let mut order: Vec<usize> = (0..nrows).collect();
        order.sort_by_key(|&i| (ptr[i + 1] - ptr[i], i));
        let mut blocks = Vec::new();
        let mut idx = Vec::new();
        let mut pos = Vec::new();
        let mut i = 0;
        while i < nrows {
            let len = ptr[order[i] + 1] - ptr[order[i]];
            let mut rows = [0usize; LANES];
            let mut nr = 0;
            while i < nrows && nr < LANES && ptr[order[i] + 1] - ptr[order[i]] == len {
                rows[nr] = order[i];
                nr += 1;
                i += 1;
            }
            let start = idx.len() / LANES;
            for k in 0..len {
                for r in 0..LANES {
                    if r < nr {
                        let p = ptr[rows[r]] + k;
                        idx.push(indices[p]);
                        pos.push(p);
                    } else {
                        idx.push(0);
                        pos.push(usize::MAX);
                    }
                }
            }
            blocks.push((start, len, nr, rows));
        }
        let ncols = idx.iter().copied().max().map_or(0, |m| m + 1);
        BlockedPattern { blocks, idx, pos, nrows, ncols }
    }
    fn gather(&self, src: &[f64], dst: &mut Vec<f64>) {
        dst.clear();
        dst.extend(self.pos.iter().map(|&p| if p == usize::MAX { 0.0 } else { src[p] }));
    }
    #[inline]
    fn matvec(&self, vals: &[f64], x: &[f64], out: &mut [f64]) {
        assert!(out.len() == self.nrows && x.len() >= self.ncols && vals.len() == self.idx.len());
        for &(start, len, nr, rows) in &self.blocks {
            let mut acc = [0.0f64; LANES];
            let mut base = start * LANES;
            for _ in 0..len {
                // SAFETY: base + LANES <= idx.len() == vals.len() by construction,
                // and every index is < ncols <= x.len() (asserted above).
                unsafe {
                    let v = vals.get_unchecked(base..base + LANES);
                    let ix = self.idx.get_unchecked(base..base + LANES);
                    for r in 0..LANES {
                        acc[r] += v[r] * *x.get_unchecked(ix[r]);
                    }
                }
                base += LANES;
            }
            for r in 0..nr {
                out[rows[r]] = acc[r];
            }
        }
    }
}

/// Sparse Jacobian with blocked copies of its CSR pattern and of its transpose,
/// so both `J @ x` and `J.T @ x` are lane-parallel gather dot products that
/// reproduce scipy's `csr_matvec` / `csc_matvec` bit for bit.
pub struct Csr<'p> {
    pub m: usize,
    pub n: usize,
    plan: &'p SparsityPlan,
    pub data: Vec<f64>,
    bdata: Vec<f64>,
    tdata: Vec<f64>,
}

impl<'p> Csr<'p> {
    fn new(plan: &'p SparsityPlan, data: Vec<f64>) -> Self {
        let mut bdata = Vec::new();
        let mut tdata = Vec::new();
        plan.blocked.gather(&data, &mut bdata);
        plan.tblocked.gather(&data, &mut tdata);
        Csr { m: plan.m, n: plan.n, plan, data, bdata, tdata }
    }
    fn matvec(&self, x: &[f64], out: &mut [f64]) {
        self.plan.blocked.matvec(&self.bdata, x, out);
    }
    fn rmatvec(&self, x: &[f64], out: &mut [f64]) {
        self.plan.tblocked.matvec(&self.tdata, x, out);
    }
}

struct SparsityPlan {
    m: usize,
    n: usize,
    #[allow(dead_code)]
    indptr: Vec<usize>,
    indices: Vec<usize>,
    groups: Vec<Vec<usize>>,     // columns per group
    group_rows: Vec<Vec<usize>>, // union of rows touched by a group (sorted)
    col_rows: Vec<Vec<usize>>,   // rows per column
    col_pos: Vec<Vec<usize>>,    // data position for each (col, row)
    // transposed (CSC) pattern: per column, rows in increasing order and the
    // position of each entry in the CSR data array
    blocked: BlockedPattern,
    tblocked: BlockedPattern,
}

fn build_plan(rows: &[Vec<(usize, usize)>], n: usize) -> SparsityPlan {
    let m = rows.len();
    let mut indptr = vec![0usize; m + 1];
    let mut indices: Vec<usize> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let mut cols: Vec<usize> = Vec::new();
        for &(s, e) in r {
            cols.extend(s..e);
        }
        cols.sort_unstable();
        cols.dedup();
        indices.extend_from_slice(&cols);
        indptr[i + 1] = indices.len();
    }
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut col_pos: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..m {
        for jj in indptr[i]..indptr[i + 1] {
            col_rows[indices[jj]].push(i);
            col_pos[indices[jj]].push(jj);
        }
    }
    // greedy column grouping (Curtis-Powell-Reid): columns with disjoint row sets share a group
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_mask: Vec<Vec<bool>> = Vec::new();
    for c in 0..n {
        let mut placed = false;
        for (g, gm) in group_mask.iter_mut().enumerate() {
            if col_rows[c].iter().all(|&r| !gm[r]) {
                for &r in &col_rows[c] {
                    gm[r] = true;
                }
                groups[g].push(c);
                placed = true;
                break;
            }
        }
        if !placed {
            let mut gm = vec![false; m];
            for &r in &col_rows[c] {
                gm[r] = true;
            }
            group_mask.push(gm);
            groups.push(vec![c]);
        }
    }
    let group_rows: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            let mut rs: Vec<usize> = g.iter().flat_map(|&c| col_rows[c].iter().cloned()).collect();
            rs.sort_unstable();
            rs.dedup();
            rs
        })
        .collect();
    // transposed pattern (CSC): per column, rows in increasing order, with each
    // entry's position in the CSR data array
    let mut tptr = vec![0usize; n + 1];
    let mut tidx: Vec<usize> = Vec::with_capacity(indices.len());
    let mut tpos: Vec<usize> = Vec::with_capacity(indices.len());
    for c in 0..n {
        tidx.extend_from_slice(&col_rows[c]);
        tpos.extend_from_slice(&col_pos[c]);
        tptr[c + 1] = tidx.len();
    }
    let blocked = BlockedPattern::new(&indptr, &indices);
    let mut tblocked = BlockedPattern::new(&tptr, &tidx);
    // tblocked.pos currently indexes the CSC arrays; redirect to CSR data positions
    for p in tblocked.pos.iter_mut() {
        if *p != usize::MAX {
            *p = tpos[*p];
        }
    }
    SparsityPlan { m, n, indptr, indices, groups, group_rows, col_rows, col_pos, blocked, tblocked }
}

struct JacWorkspace {
    pc: Vec<PC>,
    x: Vec<f64>,
    frow: Vec<f64>,
    row_lookup: Vec<usize>,
}

/// `approx_derivative(..., method='2-point', sparsity=...)`: J[r, c] = (f(x + h e_group) - f0)[r] / dx_c.
fn sparse_jacobian<'p>(problem: &Problem, x0: &[f64], f0: &[f64], plan: &'p SparsityPlan, ws: &mut JacWorkspace) -> Csr<'p> {
    let rstep = (f64::EPSILON).sqrt();
    let h: Vec<f64> = x0.iter().map(|&xi| rstep * (if xi >= 0.0 { 1.0 } else { -1.0 }) * 1.0f64.max(xi.abs())).collect();
    let mut data = vec![0.0; plan.indices.len()];
    ws.x.clear();
    ws.x.extend_from_slice(x0);
    if ws.row_lookup.len() < plan.m {
        ws.row_lookup.resize(plan.m, 0);
    }
    for (gi, group) in plan.groups.iter().enumerate() {
        let rows = &plan.group_rows[gi];
        ws.x.copy_from_slice(x0);
        for &c in group {
            ws.x[c] = x0[c] + h[c];
        }
        if ws.frow.len() < rows.len() {
            ws.frow.resize(rows.len(), 0.0);
        }
        problem.residuals_rows(&ws.x, rows, &mut ws.pc, &mut ws.frow[..rows.len()]);
        for (k, &r) in rows.iter().enumerate() {
            ws.row_lookup[r] = k;
        }
        for &c in group {
            let dx = (x0[c] + h[c]) - x0[c];
            for (k, &r) in plan.col_rows[c].iter().enumerate() {
                let df = ws.frow[ws.row_lookup[r]] - f0[r];
                data[plan.col_pos[c][k]] = df / dx;
            }
        }
    }
    Csr::new(plan, data)
}

fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    fn sign(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    if b == 0.0 {
        (sign(a), 0.0, a.abs())
    } else if a == 0.0 {
        (0.0, sign(b), b.abs())
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = sign(b) / (1.0 + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c = sign(a) / (1.0 + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

/// Operator J_h = J diag(d), with scratch buffers.
struct JH<'a> {
    j: &'a Csr<'a>,
    d: &'a [f64],
    xd: Vec<f64>,
}
impl<'a> JH<'a> {
    fn new(j: &'a Csr<'a>, d: &'a [f64]) -> Self {
        JH { j, d, xd: vec![0.0; j.n] }
    }
    fn matvec(&mut self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.j.n {
            self.xd[i] = x[i] * self.d[i];
        }
        self.j.matvec(&self.xd, out);
    }
    fn rmatvec(&self, x: &[f64], out: &mut [f64]) {
        self.j.rmatvec(x, out);
        for i in 0..out.len() {
            out[i] = self.d[i] * out[i];
        }
    }
    fn matvec_alloc(&mut self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.j.m];
        self.matvec(x, &mut out);
        out
    }
}

/// scipy.sparse.linalg.lsmr with default tolerances; returns x.
fn lsmr(a: &mut JH, b: &[f64], damp: f64) -> Vec<f64> {
    let (atol, btol, conlim) = (1e-6, 1e-6, 1e8);
    let m = a.j.m;
    let n = a.j.n;
    let maxiter = m.min(n);
    let mut u = b.to_vec();
    let normb = bnorm(b);
    let mut x = vec![0.0; n];
    let mut beta = normb;
    let mut v = vec![0.0; n];
    let mut alpha;
    let mut tmp_m = vec![0.0; m];
    let mut tmp_n = vec![0.0; n];
    if beta > 0.0 {
        let s = 1.0 / beta;
        for ui in u.iter_mut() {
            *ui *= s;
        }
        a.rmatvec(&u, &mut v);
        alpha = bnorm(&v);
    } else {
        alpha = 0.0;
    }
    if alpha > 0.0 {
        let s = 1.0 / alpha;
        for vi in v.iter_mut() {
            *vi *= s;
        }
    }
    let mut itn = 0;
    let mut zetabar = alpha * beta;
    let mut alphabar = alpha;
    let mut rho = 1.0;
    let mut rhobar = 1.0;
    let mut cbar = 1.0;
    let mut sbar = 0.0;
    let mut h = v.clone();
    let mut hbar = vec![0.0; n];
    let mut betadd = beta;
    let mut betad = 0.0;
    let mut rhodold = 1.0;
    let mut tautildeold = 0.0;
    let mut thetatilde = 0.0;
    let mut zeta = 0.0;
    let mut d = 0.0;
    let mut norma2 = alpha * alpha;
    let mut maxrbar = 0.0f64;
    let mut minrbar = 1e100f64;
    let ctol = if conlim > 0.0 { 1.0 / conlim } else { 0.0 };
    let normar0 = alpha * beta;
    if normar0 == 0.0 || normb == 0.0 {
        return x;
    }
    let mut sw_mv = crate::Stopwatch::new();
    let mut sw_rmv = crate::Stopwatch::new();
    let mut sw_nrm = crate::Stopwatch::new();
    let mut sw_rest = crate::Stopwatch::new();
    while itn < maxiter {
        itn += 1;
        sw_mv.start();
        a.matvec(&v, &mut tmp_m);
        sw_mv.stop();
        for i in 0..m {
            u[i] = u[i] * (-alpha) + tmp_m[i];
        }
        sw_nrm.start();
        beta = bnorm(&u);
        sw_nrm.stop();
        if beta > 0.0 {
            let s = 1.0 / beta;
            for ui in u.iter_mut() {
                *ui *= s;
            }
            sw_rmv.start();
            a.rmatvec(&u, &mut tmp_n);
            sw_rmv.stop();
            for i in 0..n {
                v[i] = v[i] * (-beta) + tmp_n[i];
            }
            sw_nrm.start();
            alpha = bnorm(&v);
            sw_nrm.stop();
            if alpha > 0.0 {
                let s = 1.0 / alpha;
                for vi in v.iter_mut() {
                    *vi *= s;
                }
            }
        }
        sw_rest.start();
        let (chat, shat, alphahat) = sym_ortho(alphabar, damp);
        let rhoold = rho;
        let (c, s, rho_new) = sym_ortho(alphahat, beta);
        rho = rho_new;
        let thetanew = s * alpha;
        alphabar = c * alpha;
        let rhobarold = rhobar;
        let zetaold = zeta;
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        let (cbar_n, sbar_n, rhobar_n) = sym_ortho(cbar * rho, thetanew);
        cbar = cbar_n;
        sbar = sbar_n;
        rhobar = rhobar_n;
        zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;
        let f1 = -(thetabar * rho / (rhoold * rhobarold));
        for i in 0..n {
            hbar[i] = hbar[i] * f1 + h[i];
        }
        let f2 = zeta / (rho * rhobar);
        for i in 0..n {
            x[i] += f2 * hbar[i];
        }
        let f3 = -(thetanew / rho);
        for i in 0..n {
            h[i] = h[i] * f3 + v[i];
        }
        let betaacute = chat * betadd;
        let betacheck = -shat * betadd;
        let betahat = c * betaacute;
        betadd = -s * betaacute;
        let thetatildeold = thetatilde;
        let (ctildeold, stildeold, rhotildeold) = sym_ortho(rhodold, thetabar);
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;
        d += betacheck * betacheck;
        let normr = (d + (betad - taud) * (betad - taud) + betadd * betadd).sqrt();
        norma2 += beta * beta;
        let norma = norma2.sqrt();
        norma2 += alpha * alpha;
        maxrbar = maxrbar.max(rhobarold);
        if itn > 1 {
            minrbar = minrbar.min(rhobarold);
        }
        let conda = maxrbar.max(rhotemp) / minrbar.min(rhotemp);
        let normar = zetabar.abs();
        sw_rest.stop();
        sw_nrm.start();
        let normx = bnorm(&x);
        sw_nrm.stop();
        let test1 = normr / normb;
        let test2 = if norma * normr != 0.0 { normar / (norma * normr) } else { f64::INFINITY };
        let test3 = 1.0 / conda;
        let t1 = test1 / (1.0 + norma * normx / normb);
        let rtol = btol + atol * norma * normx / normb;
        let mut istop = 0;
        if itn >= maxiter {
            istop = 7;
        }
        if 1.0 + test3 <= 1.0 {
            istop = 6;
        }
        if 1.0 + test2 <= 1.0 {
            istop = 5;
        }
        if 1.0 + t1 <= 1.0 {
            istop = 4;
        }
        if test3 <= ctol {
            istop = 3;
        }
        if test2 <= atol {
            istop = 2;
        }
        if test1 <= rtol {
            istop = 1;
        }
        if istop > 0 {
            break;
        }
    }
    if crate::timing_enabled() {
        eprintln!(
            "        lsmr itn={} mv={} rmv={} nrm={} rest={}",
            itn,
            sw_mv.report(),
            sw_rmv.report(),
            sw_nrm.report(),
            sw_rest.report()
        );
    }
    x
}

/// Economic QR of an n x 2 matrix given as two columns; returns the two
/// orthonormal columns of Q (LAPACK dgeqr2 + dorg2r conventions).
fn qr_econ_2(c0: &[f64], c1: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = c0.len();
    let mut a0 = c0.to_vec();
    let mut a1 = c1.to_vec();
    let (tau1, v1) = {
        let alpha = a0[0];
        let xnorm = crate::numeric::dnrm2(&a0[1..]);
        if xnorm == 0.0 {
            (0.0, a0.clone())
        } else {
            let beta = -(alpha.lhypot(xnorm)).copysign(alpha);
            let tau = (beta - alpha) / beta;
            let sc = 1.0 / (alpha - beta);
            let mut v = a0.clone();
            v[0] = 1.0;
            for i in 1..n {
                v[i] *= sc;
            }
            a0[0] = beta;
            (tau, v)
        }
    };
    if tau1 != 0.0 {
        let mut w = 0.0;
        for i in 0..n {
            w += v1[i] * a1[i];
        }
        for i in 0..n {
            a1[i] -= tau1 * w * v1[i];
        }
    }
    let (tau2, v2) = {
        if n < 2 {
            (0.0, vec![0.0; n])
        } else {
            let alpha = a1[1];
            let xnorm = crate::numeric::dnrm2(&a1[2..]);
            if xnorm == 0.0 {
                let mut v = vec![0.0; n];
                v[1] = 1.0;
                (0.0, v)
            } else {
                let beta = -(alpha.lhypot(xnorm)).copysign(alpha);
                let tau = (beta - alpha) / beta;
                let sc = 1.0 / (alpha - beta);
                let mut v = vec![0.0; n];
                v[1] = 1.0;
                for i in 2..n {
                    v[i] = a1[i] * sc;
                }
                (tau, v)
            }
        }
    };
    let mut q1 = vec![0.0; n];
    q1[1] = 1.0;
    if tau2 != 0.0 {
        let mut w = 0.0;
        for i in 1..n {
            w += v2[i] * q1[i];
        }
        for i in 1..n {
            q1[i] -= tau2 * w * v2[i];
        }
    }
    if tau1 != 0.0 {
        let mut w = 0.0;
        for i in 0..n {
            w += v1[i] * q1[i];
        }
        for i in 0..n {
            q1[i] -= tau1 * w * v1[i];
        }
    }
    let mut q0 = vec![0.0; n];
    q0[0] = 1.0;
    if tau1 != 0.0 {
        let mut w = 0.0;
        for i in 0..n {
            w += v1[i] * q0[i];
        }
        for i in 0..n {
            q0[i] -= tau1 * w * v1[i];
        }
    }
    (q0, q1)
}

fn solve_trust_region_2d(b: [[f64; 2]; 2], g: [f64; 2], delta: f64) -> [f64; 2] {
    let mut newton: Option<[f64; 2]> = None;
    if b[0][0] > 0.0 {
        let r00 = b[0][0].sqrt();
        let r01 = b[0][1] / r00;
        let t = b[1][1] - r01 * r01;
        if t > 0.0 {
            let r11 = t.sqrt();
            let y0 = g[0] / r00;
            let y1 = (g[1] - r01 * y0) / r11;
            let p1 = y1 / r11;
            let p0 = (y0 - r01 * p1) / r00;
            let p = [-p0, -p1];
            if dot2(p, p) <= delta * delta {
                newton = Some(p);
            }
        }
    }
    if let Some(p) = newton {
        return p;
    }
    let a = b[0][0] * delta * delta;
    let bb = b[0][1] * delta * delta;
    let c = b[1][1] * delta * delta;
    let d = g[0] * delta;
    let f = g[1] * delta;
    let coeffs = [-bb + d, 2.0 * (a - c + f), 6.0 * bb, 2.0 * (-a + c + f), -bb - d];
    let ts = real_roots(&coeffs);
    let mut best_val = f64::INFINITY;
    let mut best = [0.0, 0.0];
    for t in ts {
        let p = [delta * (2.0 * t / (1.0 + t * t)), delta * ((1.0 - t * t) / (1.0 + t * t))];
        let bp = [b[0][0] * p[0] + b[0][1] * p[1], b[1][0] * p[0] + b[1][1] * p[1]];
        let value = 0.5 * (p[0] * bp[0] + p[1] * bp[1]) + (g[0] * p[0] + g[1] * p[1]);
        if value < best_val {
            best_val = value;
            best = p;
        }
    }
    best
}

/// `least_squares(f, x0, method='trf', x_scale=x_scale, max_nfev=max_nfev, ftol=xtol=gtol=1e-7, jac_sparsity=...)`
pub fn trf_solve(problem: &Problem, x0: &[f64], x_scale: &[f64], max_nfev: usize) -> Vec<f64> {
    let (ftol, xtol, gtol) = (1e-7, 1e-7, 1e-7);
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut t_jac = crate::Stopwatch::new();
    let mut t_lsmr = crate::Stopwatch::new();
    let mut t_res = crate::Stopwatch::new();
    let mut t_all = crate::Stopwatch::new();
    t_all.start();
    let mut f = problem.residuals(&x);
    let m = f.len();
    let plan = build_plan(&problem.row_cols, n);
    let mut ws = JacWorkspace { pc: Vec::new(), x: Vec::new(), frow: Vec::new(), row_lookup: Vec::new() };
    let mut jac = sparse_jacobian(problem, &x, &f, &plan, &mut ws);
    let mut nfev = 1usize;
    let mut cost = 0.5 * ddot_self(&f);
    let mut g = vec![0.0; n];
    jac.rmatvec(&f, &mut g);
    let scale = x_scale.to_vec();
    let scale_inv: Vec<f64> = scale.iter().map(|s| 1.0 / s).collect();
    let xs: Vec<f64> = x.iter().zip(&scale_inv).map(|(a, b)| a * b).collect();
    let mut delta = bnorm(&xs);
    if delta == 0.0 {
        delta = 1.0;
    }
    let mut termination: Option<i32> = None;
    let mut js = vec![0.0; m];
    loop {
        let g_norm = g.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if g_norm < gtol {
            termination = Some(1);
        }
        if termination.is_some() || nfev == max_nfev {
            break;
        }
        let d = &scale;
        let g_h: Vec<f64> = d.iter().zip(&g).map(|(a, b)| a * b).collect();
        let mut jh = JH::new(&jac, d);
        // regularization term (build_quadratic_1d + minimize_quadratic_1d)
        let reg_term = {
            let s: Vec<f64> = g_h.iter().map(|v| -v).collect();
            let v = jh.matvec_alloc(&s);
            let mut a = ddot_self(&v);
            a *= 0.5;
            let b = ddot(&g_h, &s);
            let to_tr = delta / bnorm(&g_h);
            let mut ts = vec![0.0, to_tr];
            if a != 0.0 {
                let extremum = -0.5 * b / a;
                if 0.0 < extremum && extremum < to_tr {
                    ts.push(extremum);
                }
            }
            let ys: Vec<f64> = ts.iter().map(|&t| t * (a * t + b)).collect();
            let mi = crate::numeric::np_argmin(&ys);
            let ag_value = ys[mi];
            -ag_value / (delta * delta)
        };
        let damp_full = (0.0f64 + reg_term).sqrt();
        t_lsmr.start();
        let gn_h = lsmr(&mut jh, &f, damp_full);
        t_lsmr.stop();
        let (s0, s1) = qr_econ_2(&g_h, &gn_h);
        let js0 = jh.matvec_alloc(&s0);
        let js1 = jh.matvec_alloc(&s1);
        let b_s = [[ddot_self(&js0), ddot(&js0, &js1)], [ddot(&js1, &js0), ddot_self(&js1)]];
        let g_s = [ddot(&s0, &g_h), ddot(&s1, &g_h)];

        let mut actual_reduction = -1.0f64;
        let mut x_new = x.clone();
        let mut f_new = f.clone();
        let mut cost_new = cost;
        while actual_reduction <= 0.0 && nfev < max_nfev {
            let p_s = solve_trust_region_2d(b_s, g_s, delta);
            let step_h: Vec<f64> = (0..n).map(|i| s0[i] * p_s[0] + s1[i] * p_s[1]).collect();
            jh.matvec(&step_h, &mut js);
            let q = ddot_self(&js);
            let l = ddot(&step_h, &g_h);
            let predicted_reduction = -(0.5 * q + l);
            let step: Vec<f64> = d.iter().zip(&step_h).map(|(a, b)| a * b).collect();
            x_new = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            t_res.start();
            f_new = problem.residuals(&x_new);
            t_res.stop();
            nfev += 1;
            let step_h_norm = bnorm(&step_h);
            if !f_new.iter().all(|v| v.is_finite()) {
                delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = 0.5 * ddot_self(&f_new);
            actual_reduction = cost - cost_new;
            let ratio = if predicted_reduction > 0.0 {
                actual_reduction / predicted_reduction
            } else if predicted_reduction == 0.0 && actual_reduction == 0.0 {
                1.0
            } else {
                0.0
            };
            let mut delta_new = delta;
            if ratio < 0.25 {
                delta_new = 0.25 * step_h_norm;
            } else if ratio > 0.75 && step_h_norm > 0.95 * delta {
                delta_new *= 2.0;
            }
            let step_norm = bnorm(&step);
            let ftol_sat = actual_reduction < ftol * cost && ratio > 0.25;
            let xtol_sat = step_norm < xtol * (xtol + bnorm(&x));
            termination = if ftol_sat && xtol_sat {
                Some(4)
            } else if ftol_sat {
                Some(2)
            } else if xtol_sat {
                Some(3)
            } else {
                None
            };
            if termination.is_some() {
                break;
            }
            delta = delta_new;
        }
        if actual_reduction > 0.0 {
            x = x_new;
            f = f_new;
            cost = cost_new;
            if termination.is_some() {
                break;
            }
            t_jac.start();
            jac = sparse_jacobian(problem, &x, &f, &plan, &mut ws);
            t_jac.stop();
            jac.rmatvec(&f, &mut g);
        }
    }
    t_all.stop();
    if crate::timing_enabled() {
        eprintln!(
            "      trf: nfev={} total={} jac={} lsmr={} residuals={}",
            nfev,
            t_all.report(),
            t_jac.report(),
            t_lsmr.report(),
            t_res.report()
        );
    }
    x
}

pub fn bbox_diag(polylines: &[Vec<P2>]) -> f64 {
    let mut any = false;
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polylines {
        for q in p {
            any = true;
            minx = minx.min(q[0]);
            miny = miny.min(q[1]);
            maxx = maxx.max(q[0]);
            maxy = maxy.max(q[1]);
        }
    }
    if !any {
        return 100.0;
    }
    norm2([maxx - minx, maxy - miny]).max(10.0)
}

pub fn solve_once(
    primitives: &[Prim],
    source_points: &[Vec<P2>],
    soft: &SoftConstraints,
    weights: Weights,
    pos_scale: f64,
    max_iters: usize,
) -> Vec<Prim> {
    if primitives.is_empty() {
        return vec![];
    }
    let x0 = pack(primitives);
    let mut initial_radii = Vec::new();
    for (i, p) in primitives.iter().enumerate() {
        match p {
            Prim::Circle { radius, .. } => initial_radii.push((i, (1e-3f64).max(*radius))),
            Prim::Arc { p0, p1, bulge } => {
                let r = arc_radius(*p0, *p1, *bulge);
                if r.is_finite() {
                    initial_radii.push((i, (1e-3f64).max(r)));
                }
            }
            _ => {}
        }
    }
    let problem = Problem::new(primitives, source_points, soft, weights, initial_radii);
    let x_scale = parameter_scales(primitives, pos_scale);
    let x = trf_solve(&problem, &x0, &x_scale, max_iters);
    unpack(&x, primitives)
}
