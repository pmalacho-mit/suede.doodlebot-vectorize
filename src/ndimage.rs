//! Ports of the `scipy.ndimage` / `skimage.draw` primitives used by the pipeline.

use crate::fmath::DetMath;
#[derive(Clone, Debug)]
pub struct Grid<T> {
    pub h: usize,
    pub w: usize,
    pub data: Vec<T>,
}

impl<T: Clone + Copy + Default> Grid<T> {
    pub fn new(h: usize, w: usize, v: T) -> Self {
        Grid { h, w, data: vec![v; h * w] }
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> T {
        self.data[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, v: T) {
        self.data[y * self.w + x] = v;
    }
    #[inline]
    pub fn idx(&self, y: usize, x: usize) -> usize {
        y * self.w + x
    }
}

pub type Mask = Grid<bool>;

impl Mask {
    pub fn any(&self) -> bool {
        self.data.iter().any(|&b| b)
    }
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }
    #[inline]
    pub fn get(&self, y: i64, x: i64) -> bool {
        if y < 0 || x < 0 || y >= self.h as i64 || x >= self.w as i64 {
            false
        } else {
            self.data[y as usize * self.w + x as usize]
        }
    }
}

pub const OFFS8: [(i64, i64); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
pub const OFFS4: [(i64, i64); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

/// `scipy.ndimage.label`: connected components numbered in raster order of
/// each component's first pixel (which is what scipy's merge-table compaction
/// produces). Returns (labels, n).
pub fn label(mask: &Mask, conn8: bool) -> (Grid<i32>, usize) {
    let (h, w) = (mask.h, mask.w);
    let mut labels = Grid::new(h, w, 0i32);
    let mut n = 0i32;
    // stack of (index, y, x) so no divisions are needed in the flood fill
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    let md = &mask.data;
    let ld = &mut labels.data;
    for sy in 0..h {
        for sx in 0..w {
            let start = sy * w + sx;
            if !md[start] || ld[start] != 0 {
                continue;
            }
            n += 1;
            ld[start] = n;
            stack.push((start, sy, sx));
            while let Some((p, y, x)) = stack.pop() {
                let up = y > 0;
                let down = y + 1 < h;
                let left = x > 0;
                let right = x + 1 < w;
                macro_rules! visit {
                    ($cond:expr, $q:expr, $ny:expr, $nx:expr) => {
                        if $cond {
                            let q = $q;
                            if md[q] && ld[q] == 0 {
                                ld[q] = n;
                                stack.push((q, $ny, $nx));
                            }
                        }
                    };
                }
                // same neighbour order as before: 4-connectivity (-1,0),(1,0),(0,-1),(0,1)
                // then the diagonals (order does not affect the labelling)
                visit!(up, p - w, y - 1, x);
                visit!(down, p + w, y + 1, x);
                visit!(left, p - 1, y, x - 1);
                visit!(right, p + 1, y, x + 1);
                if conn8 {
                    visit!(up && left, p - w - 1, y - 1, x - 1);
                    visit!(up && right, p - w + 1, y - 1, x + 1);
                    visit!(down && left, p + w - 1, y + 1, x - 1);
                    visit!(down && right, p + w + 1, y + 1, x + 1);
                }
            }
        }
    }
    (labels, n as usize)
}

/// Exact Euclidean distance transform (`scipy.ndimage.distance_transform_edt`):
/// for every True pixel, the distance to the nearest False pixel; 0 on False pixels.
pub fn distance_transform_edt(mask: &Mask) -> Grid<f64> {
    let (h, w) = (mask.h, mask.w);
    // Squared distances are kept in i32 (exact for images up to ~46k px a side;
    // larger vertical gaps saturate, which cannot change any pipeline result
    // because the row pass only needs them to be "far").
    let inf: i32 = i32::MAX;
    let sq = |d: i64| -> i32 { (d * d).min(inf as i64 - 1) as i32 };
    // squared distance along columns (1-D pass vertically): for each pixel, squared
    // distance to the nearest background pixel in the same column.
    let mut g = vec![inf; h * w];
    // Vertical 1-D pass, swept row by row (contiguous memory access).
    let none: i64 = i64::MIN;
    let mut last = vec![none; w];
    for y in 0..h {
        let row = &mut g[y * w..(y + 1) * w];
        let mrow = &mask.data[y * w..(y + 1) * w];
        for x in 0..w {
            if !mrow[x] {
                last[x] = y as i64;
                row[x] = 0;
            } else if last[x] != none {
                row[x] = sq(y as i64 - last[x]);
            }
        }
    }
    let mut last = vec![none; w];
    for y in (0..h).rev() {
        let row = &mut g[y * w..(y + 1) * w];
        let mrow = &mask.data[y * w..(y + 1) * w];
        for x in 0..w {
            if !mrow[x] {
                last[x] = y as i64;
            } else if last[x] != none {
                let v = sq(last[x] - y as i64);
                if v < row[x] {
                    row[x] = v;
                }
            }
        }
    }
    // Felzenszwalb-Huttenlocher along rows, run separately on every maximal run
    // of foreground pixels: a background column bounding the run (f = 0) dominates
    // every site beyond it, so the result is the exact EDT.
    let mut out = Grid::new(h, w, 0.0f64);
    let mut v: Vec<usize> = Vec::with_capacity(w);
    let mut z: Vec<f64> = vec![0.0; w + 2];
    let mut sites: Vec<usize> = Vec::with_capacity(w);
    for y in 0..h {
        let f = &g[y * w..(y + 1) * w];
        let mrow = &mask.data[y * w..(y + 1) * w];
        let orow = &mut out.data[y * w..(y + 1) * w];
        let mut x = 0;
        while x < w {
            if !mrow[x] {
                orow[x] = 0.0;
                x += 1;
                continue;
            }
            let x0 = x;
            while x < w && mrow[x] {
                x += 1;
            }
            let x1 = x; // exclusive
            sites.clear();
            if x0 > 0 {
                sites.push(x0 - 1);
            }
            for q in x0..x1 {
                if f[q] < inf {
                    sites.push(q);
                }
            }
            if x1 < w {
                sites.push(x1);
            }
            if sites.is_empty() {
                for q in x0..x1 {
                    orow[q] = f64::INFINITY;
                }
                continue;
            }
            v.clear();
            let mut k = 0usize;
            v.push(sites[0]);
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            for &q in &sites[1..] {
                loop {
                    let vk = v[k];
                    let fq = f[q] as f64;
                    let fv = f[vk] as f64;
                    let s = ((fq + (q * q) as f64) - (fv + (vk * vk) as f64)) / (2.0 * (q as f64 - vk as f64));
                    if s <= z[k] {
                        if k == 0 {
                            v[0] = q;
                            z[0] = f64::NEG_INFINITY;
                            z[1] = f64::INFINITY;
                            break;
                        }
                        k -= 1;
                        v.truncate(k + 1);
                        continue;
                    }
                    k += 1;
                    v.push(q);
                    z[k] = s;
                    z[k + 1] = f64::INFINITY;
                    break;
                }
            }
            let mut k = 0usize;
            for q in x0..x1 {
                while z[k + 1] < q as f64 {
                    k += 1;
                }
                let vk = v[k];
                let dx = q as i64 - vk as i64;
                let d2 = dx * dx + f[vk] as i64;
                orow[q] = if d2 >= inf as i64 { f64::INFINITY } else { (d2 as f64).sqrt() };
            }
        }
    }
    out
}

/// City-block (L1) distance to the nearest True pixel (exact, two-pass chamfer).
fn l1_distance_to_true(mask: &Mask) -> Vec<i32> {
    let (h, w) = (mask.h, mask.w);
    let big = i32::MAX / 4;
    let mut d = vec![big; h * w];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask.data[i] {
                d[i] = 0;
                continue;
            }
            let mut best = big;
            if y > 0 {
                best = best.min(d[i - w] + 1);
            }
            if x > 0 {
                best = best.min(d[i - 1] + 1);
            }
            d[i] = best;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut best = d[i];
            if y + 1 < h {
                best = best.min(d[i + w] + 1);
            }
            if x + 1 < w {
                best = best.min(d[i + 1] + 1);
            }
            d[i] = best;
        }
    }
    d
}

/// `scipy.ndimage.binary_dilation(mask, iterations=k)` with the default
/// cross-shaped structuring element and border_value=0.
pub fn binary_dilation_cross(mask: &Mask, iterations: usize) -> Mask {
    let d = l1_distance_to_true(mask);
    let mut out = Mask::new(mask.h, mask.w, false);
    for i in 0..d.len() {
        out.data[i] = d[i] as usize <= iterations;
    }
    out
}

/// `scipy.ndimage.binary_erosion(mask, structure=ones((3,3)))` with border_value=0.
pub fn binary_erosion_3x3(mask: &Mask) -> Mask {
    let (h, w) = (mask.h, mask.w);
    let mut out = Mask::new(h, w, false);
    for y in 0..h {
        for x in 0..w {
            if !mask.at(y, x) {
                continue;
            }
            let mut ok = true;
            'outer: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if !mask.get(y as i64 + dy, x as i64 + dx) {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            out.set(y, x, ok);
        }
    }
    out
}

/// scipy's running-sum 1-D uniform filter with mode='constant', cval=0.
fn uniform_filter1d_line(line: &[f64], size: usize, out: &mut [f64]) {
    let n = line.len();
    let size1 = size / 2;
    let size2 = size - size1 - 1;
    // extended line: size1 zeros, line, size2 zeros
    let ext_len = size1 + n + size2;
    let get = |i: usize| -> f64 {
        if i < size1 || i >= size1 + n {
            0.0
        } else {
            line[i - size1]
        }
    };
    let _ = ext_len;
    let mut tmp = 0.0;
    for ll in 0..size {
        tmp += get(ll);
    }
    out[0] = tmp / size as f64;
    let mut l1 = 0usize;
    let mut l2 = size;
    for ll in 1..n {
        tmp += get(l2) - get(l1);
        l2 += 1;
        l1 += 1;
        out[ll] = tmp / size as f64;
    }
}

/// `scipy.ndimage.uniform_filter(img, size, mode='constant', cval=0.0)` on a 2-D float grid.
pub fn uniform_filter(img: &Grid<f64>, size: usize) -> Grid<f64> {
    let mut out = Grid::new(img.h, img.w, 0.0f64);
    uniform_filter_into(img, size, &mut out);
    out
}

/// `uniform_filter` writing into a caller-provided grid of the same shape.
pub fn uniform_filter_into(img: &Grid<f64>, size: usize, tmp: &mut Grid<f64>) {
    let (h, w) = (img.h, img.w);
    assert!(tmp.h == h && tmp.w == w);
    // axis 0 (columns): process bands of columns together so memory access is
    // row-contiguous; each column's running sum is still evaluated in exactly
    // the scipy order (same additions, same subtractions).
    const B: usize = 64;
    let size1 = size / 2;
    let inv_n = |v: f64| v / size as f64;
    let mut x0 = 0;
    while x0 < w {
        let nb = B.min(w - x0);
        let mut acc = [0.0f64; B];
        // initial window: ext indices 0..size => line indices -size1 .. size-size1
        for ll in 0..size {
            if ll >= size1 && ll - size1 < h {
                let row = &img.data[(ll - size1) * w + x0..(ll - size1) * w + x0 + nb];
                for c in 0..nb {
                    acc[c] += row[c];
                }
            } else {
                for c in 0..nb {
                    acc[c] += 0.0;
                }
            }
        }
        {
            let out = &mut tmp.data[x0..x0 + nb];
            for c in 0..nb {
                out[c] = inv_n(acc[c]);
            }
        }
        let mut l1 = 0usize; // ext index leaving
        let mut l2 = size; // ext index entering
        for y in 1..h {
            // tmp += get(l2) - get(l1)
            let in_l2 = l2 >= size1 && l2 - size1 < h;
            let in_l1 = l1 >= size1 && l1 - size1 < h;
            match (in_l2, in_l1) {
                (true, true) => {
                    let r2 = &img.data[(l2 - size1) * w + x0..(l2 - size1) * w + x0 + nb];
                    let r1 = &img.data[(l1 - size1) * w + x0..(l1 - size1) * w + x0 + nb];
                    for c in 0..nb {
                        acc[c] += r2[c] - r1[c];
                    }
                }
                (true, false) => {
                    let r2 = &img.data[(l2 - size1) * w + x0..(l2 - size1) * w + x0 + nb];
                    for c in 0..nb {
                        acc[c] += r2[c] - 0.0;
                    }
                }
                (false, true) => {
                    let r1 = &img.data[(l1 - size1) * w + x0..(l1 - size1) * w + x0 + nb];
                    for c in 0..nb {
                        acc[c] += 0.0 - r1[c];
                    }
                }
                (false, false) => {
                    for c in 0..nb {
                        acc[c] += 0.0 - 0.0;
                    }
                }
            }
            l2 += 1;
            l1 += 1;
            let out = &mut tmp.data[y * w + x0..y * w + x0 + nb];
            for c in 0..nb {
                out[c] = inv_n(acc[c]);
            }
        }
        x0 += nb;
    }
    // axis 1 (rows), in place
    let mut outl = vec![0.0; w];
    for y in 0..h {
        uniform_filter1d_line(&tmp.data[y * w..(y + 1) * w], size, &mut outl);
        tmp.data[y * w..(y + 1) * w].copy_from_slice(&outl);
    }
}

/// `scipy.ndimage.gaussian_filter1d(x, sigma)` (order 0, mode 'reflect', truncate 4.0).
pub fn gaussian_filter1d(x: &[f64], sigma: f64) -> Vec<f64> {
    let lw = (4.0 * sigma + 0.5) as i64 as usize;
    let sigma2 = sigma * sigma;
    let mut phi: Vec<f64> = Vec::with_capacity(2 * lw + 1);
    for i in 0..(2 * lw + 1) {
        let xi = i as i64 - lw as i64;
        let v = (-0.5 / sigma2 * ((xi * xi) as f64)).lexp();
        phi.push(v);
    }
    let s = crate::numeric::np_sum(&phi);
    for v in phi.iter_mut() {
        *v /= s;
    }
    // kernel reversed (symmetric anyway); correlate1d with reflect mode.
    let n = x.len();
    let size1 = lw;
    let ext = |i: i64| -> f64 {
        // 'reflect' mode: (d c b a | a b c d | d c b a)
        let len = n as i64;
        if len == 1 {
            return x[0];
        }
        let mut j = i;
        loop {
            if j < 0 {
                j = -j - 1;
            } else if j >= len {
                j = 2 * len - j - 1;
            } else {
                break;
            }
        }
        x[j as usize]
    };
    let mut out = vec![0.0; n];
    for ll in 0..n {
        let c = ll as i64;
        let mut acc = ext(c) * phi[size1];
        let mut jj: i64 = -(size1 as i64);
        while jj < 0 {
            acc += (ext(c + jj) + ext(c - jj)) * phi[(size1 as i64 + jj) as usize];
            jj += 1;
        }
        out[ll] = acc;
    }
    out
}

/// `skimage.draw.line(r0, c0, r1, c1)` -> pixel coordinates (row, col).
pub fn bresenham(r0: i64, c0: i64, r1: i64, c1: i64) -> Vec<(i64, i64)> {
    let mut steep = false;
    let mut r = r0;
    let mut c = c0;
    let mut dr = (r1 - r0).abs();
    let mut dc = (c1 - c0).abs();
    let mut sc = if c1 - c > 0 { 1 } else { -1 };
    let mut sr = if r1 - r > 0 { 1 } else { -1 };
    if dr > dc {
        steep = true;
        std::mem::swap(&mut c, &mut r);
        std::mem::swap(&mut dc, &mut dr);
        std::mem::swap(&mut sc, &mut sr);
    }
    let mut d = 2 * dr - dc;
    let n = dc as usize + 1;
    let mut rr = vec![0i64; n];
    let mut cc = vec![0i64; n];
    for i in 0..dc as usize {
        if steep {
            rr[i] = c;
            cc[i] = r;
        } else {
            rr[i] = r;
            cc[i] = c;
        }
        while d >= 0 {
            r += sr;
            d -= 2 * dc;
        }
        c += sc;
        d += 2 * dr;
    }
    rr[dc as usize] = r1;
    cc[dc as usize] = c1;
    rr.into_iter().zip(cc).collect()
}

/// `scipy.signal.find_peaks(x, height=h, distance=d)` -> peak indices.
pub fn find_peaks(x: &[f64], height: f64, distance: f64) -> Vec<usize> {
    // local maxima (plateau midpoints)
    let n = x.len();
    let mut peaks: Vec<usize> = Vec::new();
    if n >= 3 {
        let mut i = 1usize;
        let i_max = n - 1;
        while i < i_max {
            if x[i - 1] < x[i] {
                let mut i_ahead = i + 1;
                while i_ahead < i_max && x[i_ahead] == x[i] {
                    i_ahead += 1;
                }
                if x[i_ahead] < x[i] {
                    let left = i;
                    let right = i_ahead - 1;
                    peaks.push((left + right) / 2);
                    i = i_ahead;
                }
            }
            i += 1;
        }
    }
    // height filter
    peaks.retain(|&p| x[p] >= height);
    // distance filter
    let m = peaks.len();
    if m == 0 {
        return peaks;
    }
    let dist = distance.ceil() as i64;
    let mut keep = vec![true; m];
    let priority: Vec<f64> = peaks.iter().map(|&p| x[p]).collect();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| priority[a].partial_cmp(&priority[b]).unwrap_or(std::cmp::Ordering::Equal));
    for ii in (0..m).rev() {
        let j = order[ii];
        if !keep[j] {
            continue;
        }
        let mut k = j as i64 - 1;
        while k >= 0 && (peaks[j] as i64 - peaks[k as usize] as i64) < dist {
            keep[k as usize] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < m && (peaks[k] as i64 - peaks[j] as i64) < dist {
            keep[k] = false;
            k += 1;
        }
    }
    peaks.into_iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| p).collect()
}
