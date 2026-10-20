//! Bit-level emulation of the BLAS kernels numpy dispatches to for the small
//! operations the pipeline uses (OpenBLAS SkylakeX/Haswell kernels, which fuse
//! multiply-adds). `np.linalg.norm(v)`, `np.dot(a, b)` and `(k,2) @ (2,)` all go
//! through these kernels, so matching their rounding keeps the port on the
//! reference's floating-point trajectory.

/// Correctly rounded fused multiply-add `a*b + c`.
///
/// On x86-64 with FMA support this is the hardware instruction; everywhere
/// else (notably wasm32, which has no fused multiply-add) it is the
/// Boldo–Melquiond emulation: exact product via Veltkamp splitting, TwoSum
/// with the addend, round-to-odd on the low parts, final rounding. Both
/// paths give identical bits.
#[inline]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(feature = "naive_fma_test")]
    {
        return a * b + c;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("fma") {
            // SAFETY: feature checked at runtime.
            return unsafe { fma_hw(a, b, c) };
        }
    }
    fma_soft(a, b, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fma_hw(a: f64, b: f64, c: f64) -> f64 {
    use std::arch::x86_64::*;
    _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
}

/// Software correctly-rounded FMA (Boldo & Melquiond, "Emulation of FMA and
/// correctly rounded sums: proved algorithms using rounding to odd").
#[inline]
pub fn fma_soft(a: f64, b: f64, c: f64) -> f64 {
    const SPLIT: f64 = 134217729.0; // 2^27 + 1
    let p = a * b;
    if !p.is_finite() || !c.is_finite() {
        return p + c;
    }
    // TwoProduct via Veltkamp splitting: p + pl == a*b exactly
    let t = SPLIT * a;
    let ah = t - (t - a);
    let al = a - ah;
    let t = SPLIT * b;
    let bh = t - (t - b);
    let bl = b - bh;
    let pl = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    // TwoSum(c, p)
    let th = c + p;
    let bb = th - c;
    let tl = (c - (th - bb)) + (p - bb);
    // round-to-odd(tl + pl)
    let s = tl + pl;
    let bb = s - tl;
    let e = (tl - (s - bb)) + (pl - bb);
    let v = if e != 0.0 && (s.to_bits() & 1) == 0 {
        let up = (e > 0.0) == (s > 0.0);
        f64::from_bits(if up { s.to_bits() + 1 } else { s.to_bits() - 1 })
    } else {
        s
    };
    th + v
}

/// `fma(a, a, c)` with a single Veltkamp split (same bits as `fma_soft(a, a, c)`).
#[inline]
pub fn fma_soft_sq(a: f64, c: f64) -> f64 {
    const SPLIT: f64 = 134217729.0;
    let p = a * a;
    if !p.is_finite() || !c.is_finite() {
        return p + c;
    }
    let t = SPLIT * a;
    let ah = t - (t - a);
    let al = a - ah;
    let pl = ((ah * ah - p) + 2.0 * (ah * al)) + al * al;
    let th = c + p;
    let bb = th - c;
    let tl = (c - (th - bb)) + (p - bb);
    let s = tl + pl;
    let bb = s - tl;
    let e = (tl - (s - bb)) + (pl - bb);
    let v = if e != 0.0 && (s.to_bits() & 1) == 0 {
        let up = (e > 0.0) == (s > 0.0);
        f64::from_bits(if up { s.to_bits() + 1 } else { s.to_bits() - 1 })
    } else {
        s
    };
    th + v
}

/// `ddot(x, x)`: identical bits, cheaper on targets without hardware FMA.
pub fn ddot_self(x: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", not(feature = "naive_fma_test")))]
    {
        if std::is_x86_feature_detected!("fma") {
            // SAFETY: feature checked at runtime.
            return unsafe { ddot_hw(x, x) };
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "naive_fma_test")))]
    {
        return simd128::ddot_simd(x, x, true);
    }
    #[cfg(feature = "naive_fma_test")]
    {
        return ddot_impl(x, x, fma);
    }
    #[allow(unreachable_code)]
    ddot_impl(x, x, |a, _b, c| fma_soft_sq(a, c))
}

/// OpenBLAS `ddot` (SkylakeX kernel): what `np.dot(x, y)` and
/// `np.linalg.norm(x)**2` compute for contiguous float64 vectors.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", not(feature = "naive_fma_test")))]
    {
        if std::is_x86_feature_detected!("fma") {
            // SAFETY: feature checked at runtime.
            return unsafe { ddot_hw(x, y) };
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "naive_fma_test")))]
    {
        return simd128::ddot_simd(x, y, false);
    }
    #[allow(unreachable_code)]
    ddot_impl(x, y, fma)
}

/// wasm32 SIMD128 version of the ddot kernel with a vectorised, branch-free
/// Boldo–Melquiond FMA. Bit-identical to the scalar kernel.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use core::arch::wasm32::*;

    #[inline(always)]
    fn two_sum(a: v128, b: v128) -> (v128, v128) {
        let s = f64x2_add(a, b);
        let bb = f64x2_sub(s, a);
        let e = f64x2_add(f64x2_sub(a, f64x2_sub(s, bb)), f64x2_sub(b, bb));
        (s, e)
    }

    /// Correctly rounded a*b + c per lane; `None` if any lane is non-finite.
    #[inline(always)]
    fn fma_v(a: v128, b: v128, c: v128, square: bool) -> Option<v128> {
        let split = f64x2_splat(134217729.0);
        let p = f64x2_mul(a, b);
        let inf = f64x2_splat(f64::INFINITY);
        let bad = v128_or(f64x2_ge(f64x2_abs(p), inf), f64x2_ge(f64x2_abs(c), inf));
        // NaN compares false, so also catch NaNs explicitly
        let nan = v128_or(f64x2_ne(p, p), f64x2_ne(c, c));
        if v128_any_true(v128_or(bad, nan)) {
            return None;
        }
        let t = f64x2_mul(split, a);
        let ah = f64x2_sub(t, f64x2_sub(t, a));
        let al = f64x2_sub(a, ah);
        let pl = if square {
            let two = f64x2_splat(2.0);
            f64x2_add(
                f64x2_add(f64x2_sub(f64x2_mul(ah, ah), p), f64x2_mul(two, f64x2_mul(ah, al))),
                f64x2_mul(al, al),
            )
        } else {
            let t = f64x2_mul(split, b);
            let bh = f64x2_sub(t, f64x2_sub(t, b));
            let bl = f64x2_sub(b, bh);
            f64x2_add(
                f64x2_add(f64x2_add(f64x2_sub(f64x2_mul(ah, bh), p), f64x2_mul(ah, bl)), f64x2_mul(al, bh)),
                f64x2_mul(al, bl),
            )
        };
        let (th, tl) = two_sum(c, p);
        let (s, e) = two_sum(tl, pl);
        // round-to-odd: if e != 0 and s is even, move s one ulp towards e
        let zero = f64x2_splat(0.0);
        let one = i64x2_splat(1);
        let bits = s; // reinterpret
        let even = i64x2_eq(v128_and(bits, one), i64x2_splat(0));
        let nz = f64x2_ne(e, zero);
        let up = v128_not(v128_xor(f64x2_gt(e, zero), f64x2_gt(s, zero)));
        let delta = v128_bitselect(one, i64x2_splat(-1), up);
        let delta = v128_and(delta, v128_and(even, nz));
        let v = i64x2_add(bits, delta);
        Some(f64x2_add(th, v))
    }

    #[inline(always)]
    unsafe fn ld(p: *const f64) -> v128 {
        v128_load(p as *const v128)
    }

    pub fn ddot_simd(x: &[f64], y: &[f64], square: bool) -> f64 {
        let n = x.len();
        let n1 = n & !15usize;
        let mut dot = 0.0f64;
        if n1 > 0 {
            let n32 = n1 & !31usize;
            // acc5[k][l] for k in 0..4, l in 0..8  ->  16 vectors of 2 lanes
            let mut acc5 = [f64x2_splat(0.0); 16];
            let mut i = 0;
            'blocks: while i < n32 {
                let mut next = [f64x2_splat(0.0); 16];
                for k in 0..4 {
                    let base = i + 8 * k;
                    for h in 0..4 {
                        let off = base + 2 * h;
                        // SAFETY: off + 1 < n32 <= n
                        let (xa, ya) = unsafe { (ld(x.as_ptr().add(off)), ld(y.as_ptr().add(off))) };
                        match fma_v(xa, ya, acc5[4 * k + h], square) {
                            Some(v) => next[4 * k + h] = v,
                            None => break 'blocks,
                        }
                    }
                }
                acc5 = next;
                i += 32;
            }
            if i < n32 {
                // non-finite data: finish this block and the rest with the scalar kernel,
                // continuing from the current accumulators
                return finish_scalar(x, y, i, &acc5, square);
            }
            // acc[k][l] = acc5[k][l] + acc5[k][l+4]
            let mut acc = [f64x2_splat(0.0); 8];
            for k in 0..4 {
                for h in 0..2 {
                    acc[2 * k + h] = f64x2_add(acc5[4 * k + h], acc5[4 * k + h + 2]);
                }
            }
            while i < n1 {
                let mut next = [f64x2_splat(0.0); 8];
                let mut ok = true;
                for k in 0..4 {
                    let base = i + 4 * k;
                    for h in 0..2 {
                        let off = base + 2 * h;
                        let (xa, ya) = unsafe { (ld(x.as_ptr().add(off)), ld(y.as_ptr().add(off))) };
                        match fma_v(xa, ya, acc[2 * k + h], square) {
                            Some(v) => next[2 * k + h] = v,
                            None => ok = false,
                        }
                    }
                }
                if !ok {
                    return finish_scalar16(x, y, i, &acc, square);
                }
                acc = next;
                i += 16;
            }
            dot = reduce(&acc);
        }
        for i in n1..n {
            dot = super::fma_soft(y[i], x[i], dot);
        }
        dot
    }

    fn reduce(acc: &[v128; 8]) -> f64 {
        // a[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l]; l in 0..4
        let a0 = f64x2_add(f64x2_add(f64x2_add(acc[0], acc[2]), acc[4]), acc[6]); // lanes 0,1
        let a1 = f64x2_add(f64x2_add(f64x2_add(acc[1], acc[3]), acc[5]), acc[7]); // lanes 2,3
        // half = [a0 + a2, a1 + a3]
        let half = f64x2_add(a0, a1);
        f64x2_extract_lane::<0>(half) + f64x2_extract_lane::<1>(half)
    }

    fn to_scalar5(acc5: &[v128; 16]) -> [[f64; 8]; 4] {
        let mut out = [[0.0; 8]; 4];
        for k in 0..4 {
            for h in 0..4 {
                out[k][2 * h] = f64x2_extract_lane::<0>(acc5[4 * k + h]);
                out[k][2 * h + 1] = f64x2_extract_lane::<1>(acc5[4 * k + h]);
            }
        }
        out
    }

    fn finish_scalar(x: &[f64], y: &[f64], mut i: usize, acc5v: &[v128; 16], square: bool) -> f64 {
        let _ = square;
        let n = x.len();
        let n1 = n & !15usize;
        let n32 = n1 & !31usize;
        let mut acc5 = to_scalar5(acc5v);
        while i < n32 {
            for k in 0..4 {
                let base = i + 8 * k;
                for l in 0..8 {
                    acc5[k][l] = super::fma_soft(x[base + l], y[base + l], acc5[k][l]);
                }
            }
            i += 32;
        }
        let mut acc = [[0.0f64; 4]; 4];
        for k in 0..4 {
            for l in 0..4 {
                acc[k][l] = acc5[k][l] + acc5[k][l + 4];
            }
        }
        scalar_tail16(x, y, i, acc)
    }

    fn finish_scalar16(x: &[f64], y: &[f64], i: usize, accv: &[v128; 8], square: bool) -> f64 {
        let _ = square;
        let mut acc = [[0.0f64; 4]; 4];
        for k in 0..4 {
            for h in 0..2 {
                acc[k][2 * h] = f64x2_extract_lane::<0>(accv[2 * k + h]);
                acc[k][2 * h + 1] = f64x2_extract_lane::<1>(accv[2 * k + h]);
            }
        }
        scalar_tail16(x, y, i, acc)
    }

    fn scalar_tail16(x: &[f64], y: &[f64], mut i: usize, mut acc: [[f64; 4]; 4]) -> f64 {
        let n = x.len();
        let n1 = n & !15usize;
        while i < n1 {
            for k in 0..4 {
                let base = i + 4 * k;
                for l in 0..4 {
                    acc[k][l] = super::fma_soft(x[base + l], y[base + l], acc[k][l]);
                }
            }
            i += 16;
        }
        let mut a = [0.0f64; 4];
        for l in 0..4 {
            a[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
        }
        let half = [a[0] + a[2], a[1] + a[3]];
        let mut dot = half[0] + half[1];
        for i in n1..n {
            dot = super::fma_soft(y[i], x[i], dot);
        }
        dot
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn ddot_hw(x: &[f64], y: &[f64]) -> f64 {
    ddot_impl(x, y, |a, b, c| unsafe { fma_hw(a, b, c) })
}

#[inline(always)]
fn ddot_impl(x: &[f64], y: &[f64], fma: impl Fn(f64, f64, f64) -> f64) -> f64 {
    let n = x.len();
    let n1 = n & !15usize;
    let mut dot = 0.0f64;
    if n1 > 0 {
        let n32 = n1 & !31usize;
        let mut acc5 = [[0.0f64; 8]; 4];
        let mut i = 0;
        while i < n32 {
            for k in 0..4 {
                let base = i + 8 * k;
                for l in 0..8 {
                    acc5[k][l] = fma(x[base + l], y[base + l], acc5[k][l]);
                }
            }
            i += 32;
        }
        let mut acc = [[0.0f64; 4]; 4];
        for k in 0..4 {
            for l in 0..4 {
                acc[k][l] = acc5[k][l] + acc5[k][l + 4];
            }
        }
        while i < n1 {
            for k in 0..4 {
                let base = i + 4 * k;
                for l in 0..4 {
                    acc[k][l] = fma(x[base + l], y[base + l], acc[k][l]);
                }
            }
            i += 16;
        }
        let mut a = [0.0f64; 4];
        for l in 0..4 {
            a[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
        }
        let half = [a[0] + a[2], a[1] + a[3]];
        dot = half[0] + half[1];
    }
    for i in n1..n {
        dot = fma(y[i], x[i], dot);
    }
    dot
}

/// `np.dot(a, b)` / `a @ b` for 2-vectors (ddot with n = 2).
#[inline]
pub fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    fma(a[1], b[1], a[0] * b[0])
}

/// `np.linalg.norm(v)` for a 2-vector: sqrt(ddot(v, v)).
#[inline]
pub fn norm2(v: [f64; 2]) -> f64 {
    dot2(v, v).sqrt()
}

/// `np.linalg.norm(v)` for a general contiguous vector.
pub fn norm(v: &[f64]) -> f64 {
    ddot_self(v).sqrt()
}

/// One output row of `(k,2) @ (2,)` (dgemv): fma(x0, n0, x1*n1) for k >= 2,
/// but the dot kernel order for k == 1.
#[inline]
pub fn gemv_row(x: [f64; 2], n: [f64; 2], k: usize) -> f64 {
    if k == 1 {
        dot2(x, n)
    } else {
        fma(x[0], n[0], x[1] * n[1])
    }
}
