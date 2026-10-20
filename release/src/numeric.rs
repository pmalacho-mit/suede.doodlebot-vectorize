//! Numeric helpers that reproduce the exact semantics of the numpy / LAPACK
//! routines the Python pipeline relies on (summation order, sign
//! conventions, tie-breaking), so that the port tracks the reference
//! implementation as closely as floating point allows.

use crate::fmath::DetMath;
pub type P2 = [f64; 2];

#[inline]
pub fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline]
pub fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline]
pub fn scale(a: P2, s: f64) -> P2 {
    [a[0] * s, a[1] * s]
}
/// `np.dot(a, b)` / `a @ b` for 1-D 2-vectors (BLAS ddot, fused).
#[inline]
pub fn dot(a: P2, b: P2) -> f64 {
    crate::blas::dot2(a, b)
}
/// Plain `a0*b0 + a1*b1` (numpy element-wise ops / einsum).
#[inline]
pub fn dot_naive(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
/// `np.linalg.norm(v)` of a 1-D 2-vector (goes through BLAS ddot: fused).
#[inline]
pub fn norm2(a: P2) -> f64 {
    crate::blas::norm2(a)
}
/// `np.linalg.norm(v)` of a 1-D 2-vector `a - b`.
#[inline]
pub fn dist(a: P2, b: P2) -> f64 {
    norm2(sub(a, b))
}
/// Row of `np.linalg.norm(x, axis=1)`: plain sqrt(x*x + y*y).
#[inline]
pub fn norm2_naive(a: P2) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}
#[inline]
pub fn dist_naive(a: P2, b: P2) -> f64 {
    norm2_naive(sub(a, b))
}

/// numpy's pairwise summation of a contiguous float64 array
/// (`np.add.reduce` over a 1-D contiguous axis).
pub fn np_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let lim = n - (n % 8);
        while i < lim {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// `arr.mean()` for a 1-D array.
pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `pts.mean(axis=0)` for an (n, 2) array: numpy reduces over the leading
/// axis with a sequential accumulation per column.
pub fn mean_axis0(pts: &[P2]) -> P2 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for p in pts {
        sx += p[0];
        sy += p[1];
    }
    let n = pts.len() as f64;
    [sx / n, sy / n]
}

/// `pts.sum(axis=0)` sequential.
pub fn sum_axis0(pts: &[P2]) -> P2 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for p in pts {
        sx += p[0];
        sy += p[1];
    }
    [sx, sy]
}

/// `np.median` of a 1-D array (average of the two middle elements for even n).
pub fn np_median(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    // Order statistics are algorithm-independent, so a selection gives exactly
    // numpy's (introselect-based) result.
    let mut v = a.to_vec();
    let cmp = |x: &f64, y: &f64| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal);
    let (_, hi, _) = v.select_nth_unstable_by(n / 2, cmp);
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        // largest element of the lower partition
        let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    }
}

/// `np.max` / `.max()` of a slice.
pub fn np_max(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &v in &a[1..] {
        if v > m || v.is_nan() {
            m = v;
        }
    }
    m
}
pub fn np_min(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &v in &a[1..] {
        if v < m || v.is_nan() {
            m = v;
        }
    }
    m
}
/// `np.argmax` (first occurrence).
pub fn np_argmax(a: &[f64]) -> usize {
    let mut bi = 0;
    for i in 1..a.len() {
        if a[i] > a[bi] {
            bi = i;
        }
    }
    bi
}
pub fn np_argmin(a: &[f64]) -> usize {
    let mut bi = 0;
    for i in 1..a.len() {
        if a[i] < a[bi] {
            bi = i;
        }
    }
    bi
}

/// Python's float floor division `a // b`.
pub fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let m = vx % wx; // C fmod semantics (Rust % on floats == fmod)
    let mut div = (vx - m) / wx;
    let mut md = m;
    if md != 0.0 {
        if (wx < 0.0) != (md < 0.0) {
            md += wx;
            div -= 1.0;
        }
    }
    let _ = md;
    let floordiv;
    if div != 0.0 {
        floordiv = div.floor();
        if div - floordiv > 0.5 {
            return floordiv + 1.0;
        }
    } else {
        floordiv = 0.0f64.copysign(vx / wx);
    }
    floordiv
}

/// Python's float modulo `a % b` (result has the sign of `b`).
pub fn py_mod(a: f64, b: f64) -> f64 {
    let mut m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0f64.copysign(b);
    }
    m
}

/// Python's `round()` (round half to even) returning f64.
pub fn py_round(x: f64) -> f64 {
    let r = x.round(); // half away from zero
    if (x - x.trunc()).abs() == 0.5 {
        // tie: round half to even
        let t = x.trunc();
        if (t as i64) % 2 == 0 {
            t
        } else {
            r
        }
    } else {
        r
    }
}

/// `np.allclose(a, b, atol)` for 2-vectors with numpy's default rtol=1e-5.
pub fn allclose2(a: P2, b: P2, atol: f64) -> bool {
    let rtol = 1e-5;
    (a[0] - b[0]).abs() <= atol + rtol * b[0].abs() && (a[1] - b[1]).abs() <= atol + rtol * b[1].abs()
}

/// `np.linspace(0, stop, num)` (num >= 2) following numpy's formula.
pub fn np_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return vec![];
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut out = Vec::with_capacity(num);
    if step == 0.0 {
        for i in 0..num {
            out.push((i as f64 / div) * delta + start);
        }
    } else {
        for i in 0..num {
            out.push(i as f64 * step + start);
        }
    }
    out[num - 1] = stop;
    out
}

/// `np.linspace(0, n-1, num, dtype=int)` -> truncation toward zero of the float values.
pub fn np_linspace_int(start: f64, stop: f64, num: usize) -> Vec<usize> {
    np_linspace(start, stop, num).into_iter().map(|v| v as i64 as usize).collect()
}

/// `np.gradient` (second-order central differences, first order at the ends).
pub fn np_gradient(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    if n < 2 {
        return out;
    }
    out[0] = y[1] - y[0];
    out[n - 1] = y[n - 1] - y[n - 2];
    for i in 1..n - 1 {
        out[i] = (y[i + 1] - y[i - 1]) / 2.0;
    }
    out
}

/// `np.convolve(a, v, mode="same")`.
pub fn np_convolve_same(a: &[f64], v: &[f64]) -> Vec<f64> {
    let n = a.len();
    let m = v.len();
    if n == 0 || m == 0 {
        return vec![];
    }
    // numpy computes full convolution then trims. If m > n it swaps roles.
    let (a, v, n, m) = if m > n { (v, a, m, n) } else { (a, v, n, m) };
    let full_len = n + m - 1;
    let mut full = vec![0.0; full_len];
    // numpy's correlate kernel: for each output k, sum over j: a[k - j] * v[j]
    // implemented as multiply-accumulate over the smaller array.
    for k in 0..full_len {
        let mut s = 0.0;
        for j in 0..m {
            if k >= j && k - j < n {
                s += a[k - j] * v[j];
            }
        }
        full[k] = s;
    }
    let out_len = n.max(m);
    let start = (full_len - out_len) / 2;
    full[start..start + out_len].to_vec()
}

/// `np.interp(x, xp, fp)` for increasing xp.
pub fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        if x == xp[0] {
            return fp[0];
        }
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // binary search for interval (numpy uses binary search with j s.t. xp[j] <= x < xp[j+1])
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let j = lo;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut res = slope * (x - xp[j]) + fp[j];
    if !res.is_finite() {
        // numpy's fallback for overflow handling
        res = slope * (x - xp[j + 1]) + fp[j + 1];
        if !res.is_finite() && fp[j] == fp[j + 1] {
            res = fp[j];
        }
    }
    res
}

/// `np.searchsorted(a, v)` (side='left').
pub fn np_searchsorted_left(a: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `np.unwrap` with default discont=pi, period=2pi.
pub fn np_unwrap(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    let mut out = p.to_vec();
    if n < 2 {
        return out;
    }
    let period = 2.0 * std::f64::consts::PI;
    let discont = std::f64::consts::PI;
    let interval_high = period / 2.0;
    let mut cum = 0.0;
    for i in 1..n {
        let dd = p[i] - p[i - 1];
        // ddmod = mod(dd + interval_high, period) - interval_high   (np.mod semantics)
        let mut ddmod = py_mod(dd + interval_high, period) - interval_high;
        if ddmod == -interval_high && dd > 0.0 {
            ddmod = interval_high;
        }
        let mut ph_correct = ddmod - dd;
        if dd.abs() < discont {
            ph_correct = 0.0;
        }
        cum += ph_correct;
        out[i] = p[i] + cum;
    }
    out
}

// ---------------------------------------------------------------------------
// LAPACK-faithful pieces

/// LAPACK `dlasv2`: SVD of the upper-triangular 2x2 matrix [f g; 0 h].
/// Returns (ssmin, ssmax, snr, csr, snl, csl).
pub fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    fn sign(a: f64, b: f64) -> f64 {
        // gfortran SIGN(a, b): |a| carrying the sign bit of b
        a.abs().copysign(b)
    }
    let eps = f64::EPSILON * 0.5; // dlamch('E')
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (mut ssmin, mut ssmax);
    let (clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut r_clt = 0.0;
        let mut r_crt = 0.0;
        let mut r_slt = 0.0;
        let mut r_srt = 0.0;
        ssmin = 0.0;
        ssmax = 0.0;
        if ga > fa {
            pmax = 2;
            if fa / ga < eps {
                gasmal = false;
                ssmax = ga;
                if ha > 1.0 {
                    ssmin = fa / (ga / ha);
                } else {
                    ssmin = fa / ga * ha;
                }
                r_clt = 1.0;
                r_slt = ht / gt;
                r_srt = 1.0;
                r_crt = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
            let a = (s + r) * 0.5;
            ssmin = ha / a;
            ssmax = fa * a;
            if mm == 0.0 {
                if l == 0.0 {
                    t = sign(2.0, ft) * sign(1.0, gt);
                } else {
                    t = gt / sign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (a + 1.0);
            }
            l = (t * t + 4.0).sqrt();
            r_crt = 2.0 / l;
            r_srt = t / l;
            r_clt = (r_crt + r_srt * m) / a;
            r_slt = ht / ft * r_srt / a;
        }
        clt = r_clt;
        crt = r_crt;
        slt = r_slt;
        srt = r_srt;
    }
    let (csl, snl, csr, snr);
    if swap {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    } else {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// Reference `dnrm2` (scaled sum of squares).
pub fn dnrm2(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &xi in x {
        if xi != 0.0 {
            let absxi = xi.abs();
            if scale < absxi {
                ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                scale = absxi;
            } else {
                ssq += (absxi / scale) * (absxi / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// `dlapy2(x, y)` = sqrt(x^2 + y^2) avoiding overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    let xa = x.abs();
    let ya = y.abs();
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// LAPACK `dlarfg` on (alpha, x): returns (beta, tau) and scales x in place to v[1..].
fn dlarfg(alpha: f64, x: &mut [f64]) -> (f64, f64) {
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let beta = -dlapy2(alpha, xnorm).copysign(alpha);
    // (safmin rescaling loop omitted: magnitudes here are far from underflow)
    let tau = (beta - alpha) / beta;
    let sc = 1.0 / (alpha - beta);
    for v in x.iter_mut() {
        *v *= sc;
    }
    (beta, tau)
}

/// Right singular vectors (rows of Vt, as numpy's `np.linalg.svd(a, full_matrices=False)[2]`)
/// of an (m x 2) matrix with m >= 2, reproducing LAPACK dgesdd's sign conventions
/// (Householder QR with dlarfg, then dbdsqr's dlasv2 step, sign fix and sort).
pub fn svd_vt_n2(a: &[P2]) -> [P2; 2] {
    let m = a.len();
    assert!(m >= 2);
    // column 1
    let mut col1: Vec<f64> = a.iter().map(|p| p[0]).collect();
    let mut col2: Vec<f64> = a.iter().map(|p| p[1]).collect();
    let alpha1 = col1[0];
    let (beta1, tau1) = {
        let (b, t) = dlarfg(alpha1, &mut col1[1..]);
        (b, t)
    };
    // apply H1 = I - tau v v^T to col2, v = [1, col1[1..]]
    if tau1 != 0.0 {
        let mut w = col2[0];
        for i in 1..m {
            w += col1[i] * col2[i];
        }
        col2[0] -= tau1 * w;
        for i in 1..m {
            col2[i] -= tau1 * w * col1[i];
        }
    }
    let r12 = col2[0];
    let r22 = if m >= 2 {
        let alpha2 = col2[1];
        let (b, _t) = dlarfg(alpha2, &mut col2[2..]);
        b
    } else {
        0.0
    };
    // Bidiagonal B = [[beta1, r12],[0, r22]] ; dbdsqr with n = 2.
    let mut d = [beta1, r22];
    let mut e = r12;
    let mut vt = [[1.0, 0.0], [0.0, 1.0]];
    let eps = f64::EPSILON * 0.5;
    let unfl = f64::MIN_POSITIVE;
    let tolmul = 10.0f64.max(100.0f64.min(eps.lpowf(-0.125)));
    let tol = tolmul * eps;
    // sminoa
    let mut thresh;
    {
        let mut sminoa = d[0].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            mu = d[1].abs() * (mu / (mu + e.abs()));
            sminoa = sminoa.min(mu);
        }
        sminoa /= (2.0f64).sqrt();
        let maxitr = 6.0;
        thresh = (tol * sminoa).max(maxitr * (2.0 * (2.0 * unfl)));
        if thresh.is_nan() {
            thresh = 0.0;
        }
    }
    if e.abs() <= thresh {
        e = 0.0;
    } else {
        let (sigmn, sigmx, sinr, cosr, _sinl, _cosl) = dlasv2(d[0], e, d[1]);
        d[0] = sigmx;
        e = 0.0;
        d[1] = sigmn;
        // DROT on rows 0 and 1 of VT (identity): row0 = c*row0 + s*row1 ; row1 = c*row1 - s*row0
        let r0 = vt[0];
        let r1 = vt[1];
        vt[0] = [cosr * r0[0] + sinr * r1[0], cosr * r0[1] + sinr * r1[1]];
        vt[1] = [cosr * r1[0] - sinr * r0[0], cosr * r1[1] - sinr * r0[1]];
    }
    let _ = e;
    // make singular values positive
    for i in 0..2 {
        if d[i] < 0.0 {
            d[i] = -d[i];
            vt[i] = [-vt[i][0], -vt[i][1]];
        }
    }
    // sort decreasing (selection sort, one pass for n=2)
    {
        // LAPACK swaps d[0], d[1] (and the rows of Vt) when d[1] > d[0];
        // only Vt is returned here.
        if d[1] > d[0] {
            vt.swap(0, 1);
        }
    }
    vt
}

/// `np.linalg.solve` for a 2x2 system (LU with partial pivoting, as dgesv).
pub fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let (mut a00, mut a01, mut a10, mut a11) = (a[0][0], a[0][1], a[1][0], a[1][1]);
    let (mut b0, mut b1) = (b[0], b[1]);
    if a10.abs() > a00.abs() {
        std::mem::swap(&mut a00, &mut a10);
        std::mem::swap(&mut a01, &mut a11);
        std::mem::swap(&mut b0, &mut b1);
    }
    if a00 == 0.0 {
        return None;
    }
    let l = a10 / a00;
    let u11 = a11 - l * a01;
    if u11 == 0.0 {
        return None;
    }
    // forward: y0 = b0, y1 = b1 - l*y0
    let y1 = b1 - l * b0;
    let x1 = y1 / u11;
    let x0 = (b0 - a01 * x1) / a00;
    Some([x0, x1])
}

/// Least squares solution of M x = rhs for an (n x 3) matrix via Householder QR
/// (numerically equivalent to `np.linalg.lstsq` for full-rank systems).
pub fn lstsq3(m: &[[f64; 3]], rhs: &[f64]) -> Option<[f64; 3]> {
    let n = m.len();
    if n < 3 {
        return None;
    }
    let mut a: Vec<[f64; 3]> = m.to_vec();
    let mut b: Vec<f64> = rhs.to_vec();
    for k in 0..3 {
        // Householder for column k rows k..n
        let mut col: Vec<f64> = (k..n).map(|i| a[i][k]).collect();
        let alpha = col[0];
        let (beta, tau) = dlarfg(alpha, &mut col[1..]);
        if tau != 0.0 {
            // v = [1, col[1..]]
            for j in (k + 1)..3 {
                let mut w = a[k][j];
                for i in (k + 1)..n {
                    w += col[i - k] * a[i][j];
                }
                a[k][j] -= tau * w;
                for i in (k + 1)..n {
                    a[i][j] -= tau * w * col[i - k];
                }
            }
            let mut w = b[k];
            for i in (k + 1)..n {
                w += col[i - k] * b[i];
            }
            b[k] -= tau * w;
            for i in (k + 1)..n {
                b[i] -= tau * w * col[i - k];
            }
        }
        a[k][k] = beta;
        for i in (k + 1)..n {
            a[i][k] = 0.0;
        }
    }
    // back substitution
    if a[2][2] == 0.0 || a[1][1] == 0.0 || a[0][0] == 0.0 {
        return None;
    }
    let x2 = b[2] / a[2][2];
    let x1 = (b[1] - a[1][2] * x2) / a[1][1];
    let x0 = (b[0] - a[0][1] * x1 - a[0][2] * x2) / a[0][0];
    Some([x0, x1, x2])
}

/// Roots of a real polynomial (coefficients highest degree first), like `np.roots`
/// but via Durand–Kerner refinement on the companion matrix eigenvalue problem.
/// Only the real roots are returned (|imag| small), which is what the caller uses.
pub fn real_roots(coeffs: &[f64]) -> Vec<f64> {
    // strip leading zeros
    let mut start = 0;
    while start < coeffs.len() && coeffs[start] == 0.0 {
        start += 1;
    }
    let c = &coeffs[start..];
    if c.len() <= 1 {
        return vec![];
    }
    // strip trailing zeros (roots at zero)
    let mut end = c.len();
    let mut zero_roots = 0;
    while end > 1 && c[end - 1] == 0.0 {
        end -= 1;
        zero_roots += 1;
    }
    let c = &c[..end];
    let deg = c.len() - 1;
    let mut roots: Vec<f64> = Vec::new();
    if deg >= 1 {
        // normalized coefficients
        let lead = c[0];
        let a: Vec<f64> = c.iter().map(|v| v / lead).collect();
        // Durand-Kerner with complex arithmetic
        let mut z: Vec<(f64, f64)> = Vec::with_capacity(deg);
        let (mut re, mut im) = (0.4, 0.9);
        for _ in 0..deg {
            z.push((re, im));
            let (nre, nim) = (re * 0.4 - im * 0.9, re * 0.9 + im * 0.4);
            re = nre;
            im = nim;
        }
        let eval = |x: (f64, f64)| -> (f64, f64) {
            let mut r = (1.0, 0.0);
            for &coef in &a[1..] {
                r = (r.0 * x.0 - r.1 * x.1 + coef, r.0 * x.1 + r.1 * x.0);
            }
            r
        };
        for _ in 0..500 {
            let mut maxd = 0.0f64;
            for i in 0..deg {
                let num = eval(z[i]);
                let mut den = (1.0, 0.0);
                for j in 0..deg {
                    if i != j {
                        let d = (z[i].0 - z[j].0, z[i].1 - z[j].1);
                        den = (den.0 * d.0 - den.1 * d.1, den.0 * d.1 + den.1 * d.0);
                    }
                }
                let dd = den.0 * den.0 + den.1 * den.1;
                if dd == 0.0 {
                    continue;
                }
                let q = ((num.0 * den.0 + num.1 * den.1) / dd, (num.1 * den.0 - num.0 * den.1) / dd);
                z[i] = (z[i].0 - q.0, z[i].1 - q.1);
                maxd = maxd.max((q.0 * q.0 + q.1 * q.1).sqrt());
            }
            if maxd < 1e-15 {
                break;
            }
        }
        // Newton polish on real candidates
        for zi in z {
            let scale_ = 1.0f64.max(zi.0.abs());
            if zi.1.abs() <= 1e-9 * scale_ {
                let mut x = zi.0;
                for _ in 0..4 {
                    let mut p = 1.0;
                    let mut dp = 0.0;
                    for &coef in &a[1..] {
                        dp = dp * x + p;
                        p = p * x + coef;
                    }
                    if dp != 0.0 {
                        x -= p / dp;
                    }
                }
                roots.push(x);
            }
        }
    }
    for _ in 0..zero_roots {
        roots.push(0.0);
    }
    roots
}

/// numpy `np.hypot`.
pub fn hypot(a: f64, b: f64) -> f64 {
    a.lhypot(b)
}
