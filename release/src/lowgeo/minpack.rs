//! Faithful port of MINPACK's `lmder` (as shipped in scipy's `__minpack.c`),
//! driven the way `scipy.optimize.least_squares(method='lm')` drives it:
//! Jacobian by forward differences (`approx_derivative`, 2-point), internal
//! column scaling (`diag=None`), `factor=100`.

use crate::fmath::DetMath;
const EPSMCH: f64 = 2.220446049250313e-16;
const DWARF: f64 = 2.2250738585072014e-308;

pub fn enorm(x: &[f64]) -> f64 {
    let n = x.len();
    let rgiant = 2.0f64.lpowf(63.5);
    let rdwarf = 0.5 / rgiant;
    let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
    let (mut x1max, mut x3max) = (0.0f64, 0.0f64);
    let agiant = rgiant / n as f64;
    for &xi in x {
        let xabs = xi.abs();
        if xabs <= rdwarf || xabs >= agiant {
            if xabs > rdwarf {
                if xabs > x1max {
                    s1 = 1.0 + s1 * (x1max / xabs).powi(2);
                    x1max = xabs;
                } else {
                    s1 += (xabs / x1max).powi(2);
                }
            } else if xabs > x3max {
                s3 = 1.0 + s3 * (x3max / xabs).powi(2);
                x3max = xabs;
            } else if xabs != 0.0 {
                s3 += (xabs / x3max).powi(2);
            }
        } else {
            s2 += xabs * xabs;
        }
    }
    if s1 != 0.0 {
        return x1max * (s1 + (s2 / x1max) / x1max).sqrt();
    }
    if s2 != 0.0 {
        if s2 >= x3max {
            return (s2 * (1.0 + (x3max / s2) * (x3max * s3))).sqrt();
        } else {
            return (x3max * ((s2 / x3max) + (x3max * s3))).sqrt();
        }
    }
    x3max * s3.sqrt()
}

/// Column-major (m x n) matrix helper: a[i + lda*j]
struct Mat {
    m: usize,
    n: usize,
    d: Vec<f64>,
}
impl Mat {
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i + self.m * j]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i + self.m * j] = v;
    }
}

fn qrfac(a: &mut Mat, ipvt: &mut [usize], rdiag: &mut [f64], acnorm: &mut [f64], wa: &mut [f64]) {
    let (m, n) = (a.m, a.n);
    for j in 0..n {
        acnorm[j] = enorm(&a.d[j * m..j * m + m]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        ipvt[j] = j;
    }
    let minmn = m.min(n);
    for j in 0..minmn {
        let mut kmax = j;
        for k in j..n {
            if rdiag[k] > rdiag[kmax] {
                kmax = k;
            }
        }
        if kmax != j {
            for i in 0..m {
                let t = a.at(i, j);
                a.set(i, j, a.at(i, kmax));
                a.set(i, kmax, t);
            }
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            ipvt.swap(j, kmax);
        }
        let mut ajnorm = enorm(&a.d[j * m + j..j * m + m]);
        if ajnorm == 0.0 {
            rdiag[j] = -ajnorm;
            continue;
        }
        if a.at(j, j) < 0.0 {
            ajnorm = -ajnorm;
        }
        for i in j..m {
            let v = a.at(i, j) / ajnorm;
            a.set(i, j, v);
        }
        a.set(j, j, a.at(j, j) + 1.0);
        if j == n - 1 {
            rdiag[j] = -ajnorm;
            continue;
        }
        for k in j + 1..n {
            let mut ssum = 0.0;
            for i in j..m {
                ssum += a.at(i, j) * a.at(i, k);
            }
            let temp = ssum / a.at(j, j);
            for i in j..m {
                let v = a.at(i, k) - temp * a.at(i, j);
                a.set(i, k, v);
            }
            if rdiag[k] != 0.0 {
                let temp = a.at(j, k) / rdiag[k];
                rdiag[k] *= (0.0f64).max(1.0 - temp * temp).sqrt();
                if 0.05 * (rdiag[k] / wa[k]).powi(2) <= EPSMCH {
                    rdiag[k] = enorm(&a.d[k * m + j + 1..k * m + m]);
                    wa[k] = rdiag[k];
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
}

fn qrsolv(n: usize, r: &mut Mat, ipvt: &[usize], diag: &[f64], qtb: &[f64], x: &mut [f64], sdiag: &mut [f64], wa: &mut [f64]) {
    for j in 0..n {
        for i in j..n {
            let v = r.at(j, i);
            r.set(i, j, v);
        }
        x[j] = r.at(j, j);
        wa[j] = qtb[j];
    }
    for j in 0..n {
        let l = ipvt[j];
        if diag[l] != 0.0 {
            for k in j..n {
                sdiag[k] = 0.0;
            }
            sdiag[j] = diag[l];
            let mut qtbpj = 0.0;
            for k in j..n {
                if sdiag[k] != 0.0 {
                    let (ccos, ssin);
                    if r.at(k, k).abs() < sdiag[k].abs() {
                        let ccotan = r.at(k, k) / sdiag[k];
                        ssin = 0.5 / (0.25 + 0.25 * ccotan.powi(2)).sqrt();
                        ccos = ssin * ccotan;
                    } else {
                        let ttan = sdiag[k] / r.at(k, k);
                        ccos = 0.5 / (0.25 + 0.25 * ttan.powi(2)).sqrt();
                        ssin = ccos * ttan;
                    }
                    let v = r.at(k, k) * ccos + ssin * sdiag[k];
                    r.set(k, k, v);
                    let temp = ccos * wa[k] + ssin * qtbpj;
                    qtbpj = -ssin * wa[k] + ccos * qtbpj;
                    wa[k] = temp;
                    if k != n - 1 {
                        for i in k + 1..n {
                            let temp = ccos * r.at(i, k) + ssin * sdiag[i];
                            sdiag[i] = -ssin * r.at(i, k) + ccos * sdiag[i];
                            r.set(i, k, temp);
                        }
                    }
                }
            }
        }
        sdiag[j] = r.at(j, j);
        r.set(j, j, x[j]);
    }
    let mut nsing = n;
    for j in 0..n {
        if sdiag[j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa[j] = 0.0;
        }
    }
    if nsing > 0 {
        for k in 1..=nsing {
            let j = nsing - k;
            let mut ssum = 0.0;
            for i in j + 1..nsing {
                ssum += r.at(i, j) * wa[i];
            }
            wa[j] = (wa[j] - ssum) / sdiag[j];
        }
    }
    for j in 0..n {
        x[ipvt[j]] = wa[j];
    }
}

#[allow(clippy::too_many_arguments)]
fn lmpar(
    n: usize,
    r: &mut Mat,
    ipvt: &[usize],
    diag: &[f64],
    qtb: &[f64],
    delta: f64,
    par: &mut f64,
    x: &mut [f64],
    sdiag: &mut [f64],
    wa1: &mut [f64],
    wa2: &mut [f64],
) {
    let mut nsing = n;
    for j in 0..n {
        wa1[j] = qtb[j];
        if r.at(j, j) == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa1[j] = 0.0;
        }
    }
    if nsing > 0 {
        for k in 1..=nsing {
            let j = nsing - k;
            wa1[j] /= r.at(j, j);
            let temp = wa1[j];
            for i in 0..j {
                wa1[i] -= r.at(i, j) * temp;
            }
        }
    }
    for j in 0..n {
        x[ipvt[j]] = wa1[j];
    }
    let mut iter = 0;
    for j in 0..n {
        wa2[j] = diag[j] * x[j];
    }
    let mut dxnorm = enorm(&wa2[..n]);
    let mut fp = dxnorm - delta;
    if fp <= 0.1 * delta {
        *par = 0.0;
        return;
    }
    let mut parl = 0.0;
    if nsing == n {
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            let mut ssum = 0.0;
            for i in 0..j {
                ssum += r.at(i, j) * wa1[i];
            }
            wa1[j] = (wa1[j] - ssum) / r.at(j, j);
        }
        let temp = enorm(&wa1[..n]);
        parl = ((fp / delta) / temp) / temp;
    }
    for j in 0..n {
        let mut ssum = 0.0;
        for i in 0..=j {
            ssum += r.at(i, j) * qtb[i];
        }
        let l = ipvt[j];
        wa1[j] = ssum / diag[l];
    }
    let gnorm = enorm(&wa1[..n]);
    let mut paru = gnorm / delta;
    if paru == 0.0 {
        paru = DWARF / delta.min(0.1);
    }
    if *par == 0.0 {
        *par = gnorm / dxnorm;
    }
    loop {
        iter += 1;
        if *par == 0.0 {
            *par = DWARF.max(paru * 0.001);
        }
        let temp = par.sqrt();
        for j in 0..n {
            wa1[j] = temp * diag[j];
        }
        qrsolv(n, r, ipvt, &wa1[..n].to_vec(), qtb, x, sdiag, wa2);
        for j in 0..n {
            wa2[j] = diag[j] * x[j];
        }
        dxnorm = enorm(&wa2[..n]);
        let temp = fp;
        fp = dxnorm - delta;
        if fp.abs() <= 0.1 * delta || (parl == 0.0 && fp <= temp && temp < 0.0) || iter == 10 {
            return;
        }
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            wa1[j] /= sdiag[j];
            let temp = wa1[j];
            if j != n - 1 {
                for i in j + 1..n {
                    wa1[i] -= r.at(i, j) * temp;
                }
            }
        }
        let temp = enorm(&wa1[..n]);
        let parc = ((fp / delta) / temp) / temp;
        if fp > 0.0 {
            parl = parl.max(*par);
        }
        if fp < 0.0 {
            paru = paru.min(*par);
        }
        *par = parl.max(*par + parc);
    }
}

/// scipy `approx_derivative(fun, x, f0, method='2-point')` (dense), returned
/// column-major (m x n).
fn approx_jacobian<F: Fn(&[f64]) -> Vec<f64>>(fun: &F, x: &[f64], f0: &[f64]) -> Mat {
    let n = x.len();
    let m = f0.len();
    let rstep = EPSMCH.sqrt();
    let mut j = Mat { m, n, d: vec![0.0; m * n] };
    let mut x1 = x.to_vec();
    for i in 0..n {
        let sign = if x[i] >= 0.0 { 1.0 } else { -1.0 };
        let h = rstep * sign * 1.0f64.max(x[i].abs());
        x1.copy_from_slice(x);
        x1[i] = x[i] + h;
        let dx = (x[i] + h) - x[i];
        let f1 = fun(&x1);
        for r in 0..m {
            j.d[r + m * i] = (f1[r] - f0[r]) / dx;
        }
    }
    j
}

/// Returns the solution x (like `least_squares(fun, x0, method='lm', max_nfev=maxfev).x`).
pub fn lm_least_squares<F: Fn(&[f64]) -> Vec<f64>>(fun: F, x0: &[f64], maxfev: usize) -> Option<Vec<f64>> {
    let (ftol, xtol, gtol, factor) = (1e-8, 1e-8, 1e-8, 100.0);
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fvec = fun(&x);
    let m = fvec.len();
    if m < n || n == 0 {
        return None;
    }
    if !fvec.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut fjac;
    let mut diag = vec![0.0; n];
    let mut qtf = vec![0.0; n];
    let mut wa1 = vec![0.0; n];
    let mut wa2 = vec![0.0; n];
    let mut wa3 = vec![0.0; n];
    let mut wa4 = vec![0.0; m];
    let mut ipvt = vec![0usize; n];
    let mut nfev = 1usize;
    let mut fnorm = enorm(&fvec);
    let mut par = 0.0;
    let mut iter = 1;
    let mut delta = 0.0;
    let mut xnorm = 0.0;
    let mut info = 0;
    'outer: loop {
        // jacobian at x (fvec is f(x))
        fjac = approx_jacobian(&fun, &x, &fvec);
        let mut acnorm = vec![0.0; n];
        qrfac(&mut fjac, &mut ipvt, &mut wa1, &mut acnorm, &mut wa3);
        wa2.copy_from_slice(&acnorm);
        if iter == 1 {
            for j in 0..n {
                diag[j] = wa2[j];
                if wa2[j] == 0.0 {
                    diag[j] = 1.0;
                }
            }
            for j in 0..n {
                wa3[j] = diag[j] * x[j];
            }
            xnorm = enorm(&wa3);
            delta = factor * xnorm;
            if delta == 0.0 {
                delta = factor;
            }
        }
        wa4.copy_from_slice(&fvec);
        for j in 0..n {
            if fjac.at(j, j) != 0.0 {
                let mut ssum = 0.0;
                for i in j..m {
                    ssum += fjac.at(i, j) * wa4[i];
                }
                let temp = -ssum / fjac.at(j, j);
                for i in j..m {
                    wa4[i] += fjac.at(i, j) * temp;
                }
            }
            fjac.set(j, j, wa1[j]);
            qtf[j] = wa4[j];
        }
        let mut gnorm = 0.0f64;
        if fnorm != 0.0 {
            for j in 0..n {
                let l = ipvt[j];
                if wa2[l] != 0.0 {
                    let mut ssum = 0.0;
                    for i in 0..=j {
                        ssum += fjac.at(i, j) * (qtf[i] / fnorm);
                    }
                    gnorm = gnorm.max((ssum / wa2[l]).abs());
                }
            }
        }
        if gnorm <= gtol {
            info = 4;
        }
        if info != 0 {
            break 'outer;
        }
        for j in 0..n {
            diag[j] = diag[j].max(wa2[j]);
        }
        loop {
            // lmpar operates on the upper triangle of fjac (as an n x n view)
            lmpar(n, &mut fjac, &ipvt, &diag, &qtf, delta, &mut par, &mut wa1, &mut wa2, &mut wa3, &mut wa4);
            for j in 0..n {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            let pnorm = enorm(&wa3);
            if iter == 1 {
                delta = delta.min(pnorm);
            }
            let fnew = fun(&wa2);
            nfev += 1;
            wa4.copy_from_slice(&fnew);
            let fnorm1 = enorm(&wa4);
            let actred = if 0.1 * fnorm1 < fnorm { 1.0 - (fnorm1 / fnorm).powi(2) } else { -1.0 };
            for j in 0..n {
                wa3[j] = 0.0;
                let l = ipvt[j];
                let temp = wa1[l];
                for i in 0..=j {
                    wa3[i] += fjac.at(i, j) * temp;
                }
            }
            let temp1 = enorm(&wa3) / fnorm;
            let temp2 = (par.sqrt() * pnorm) / fnorm;
            let prered = temp1 * temp1 + temp2 * temp2 / 0.5;
            let dirder = -(temp1 * temp1 + temp2 * temp2);
            let ratio = if prered != 0.0 { actred / prered } else { 0.0 };
            if ratio <= 0.25 {
                let mut temp = if actred < 0.0 { (0.5 * dirder) / (dirder + 0.5 * actred) } else { 0.5 };
                if 0.1 * fnorm1 >= fnorm || temp < 0.1 {
                    temp = 0.1;
                }
                delta = temp * delta.min(pnorm / 0.1);
                par /= temp;
            } else if par == 0.0 || ratio >= 0.75 {
                delta = pnorm / 0.5;
                par *= 0.5;
            }
            if ratio >= 0.0001 {
                for j in 0..n {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                }
                fvec.copy_from_slice(&wa4);
                xnorm = enorm(&wa2);
                fnorm = fnorm1;
                iter += 1;
            }
            if actred.abs() <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 {
                info = 1;
            }
            if delta <= xtol * xnorm {
                info = 2;
            }
            if actred.abs() <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 && info == 2 {
                info = 3;
            }
            if info != 0 {
                break 'outer;
            }
            if nfev >= maxfev {
                info = 5;
            }
            if actred.abs() <= EPSMCH && prered <= EPSMCH && 0.5 * ratio <= 1.0 {
                info = 6;
            }
            if delta <= EPSMCH * xnorm {
                info = 7;
            }
            if gnorm <= EPSMCH {
                info = 8;
            }
            if info != 0 {
                break 'outer;
            }
            if ratio >= 0.0001 {
                break;
            }
        }
    }
    let _ = fjac;
    Some(x)
}
