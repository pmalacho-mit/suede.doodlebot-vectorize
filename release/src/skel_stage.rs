//! Skeletonization stage: thinning, hole collapse (`cleanup.py`), filled-region
//! resolution (`eyes.py`) and ribbon-collapse crossing resolution (`crossings.py`).

use crate::config::Config;
use crate::ndimage::{
    binary_dilation_cross, binary_erosion_3x3, bresenham, distance_transform_edt, label, uniform_filter_into, Grid, Mask,
};
use crate::numeric::{np_median, np_sum, P2};
use crate::skeleton::{skeletonize_lee, skeletonize_zhang};

// ---------------------------------------------------------------------------
// cleanup.py

fn region_max_thickness(mask: &Mask, pixels: &[usize]) -> f64 {
    // EDT of the region restricted to its bounding box padded by one background ring.
    let w = mask.w;
    let mut y0 = usize::MAX;
    let mut y1 = 0;
    let mut x0 = usize::MAX;
    let mut x1 = 0;
    for &p in pixels {
        let y = p / w;
        let x = p % w;
        y0 = y0.min(y);
        y1 = y1.max(y);
        x0 = x0.min(x);
        x1 = x1.max(x);
    }
    let bh = y1 - y0 + 3;
    let bw = x1 - x0 + 3;
    let mut sub = Mask::new(bh, bw, false);
    for &p in pixels {
        let y = p / w;
        let x = p % w;
        sub.set(y - y0 + 1, x - x0 + 1, true);
    }
    let dt = distance_transform_edt(&sub);
    let mut m = 0.0f64;
    for v in dt.data {
        if v > m {
            m = v;
        }
    }
    m
}

pub fn collapse_small_holes(skel: &Mask, cfg: &Config) -> Mask {
    let (h, w) = (skel.h, skel.w);
    // background 4-connected components
    let mut background = Mask::new(h, w, false);
    for i in 0..h * w {
        background.data[i] = !skel.data[i];
    }
    let (labels, n) = label(&background, false);
    if n == 0 {
        return skel.clone();
    }
    let mut border = vec![false; n + 1];
    for x in 0..w {
        border[labels.at(0, x) as usize] = true;
        border[labels.at(h - 1, x) as usize] = true;
    }
    for y in 0..h {
        border[labels.at(y, 0) as usize] = true;
        border[labels.at(y, w - 1) as usize] = true;
    }
    // gather pixels per label
    let mut pixels: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for i in 0..h * w {
        let l = labels.data[i] as usize;
        if l != 0 && !border[l] {
            pixels[l].push(i);
        }
    }
    let mut filled = skel.clone();
    let mut any = false;
    for l in 1..=n {
        if border[l] || pixels[l].is_empty() {
            continue;
        }
        let area = pixels[l].len() as i64;
        let mut passes = area <= cfg.max_hole_area;
        if !passes && region_max_thickness(skel, &pixels[l]) <= cfg.max_thin_thickness {
            passes = true;
        }
        if passes {
            any = true;
            for &p in &pixels[l] {
                filled.data[p] = true;
            }
        }
    }
    if any {
        filled = skeletonize_lee(&filled);
    }
    filled
}

// ---------------------------------------------------------------------------
// eyes.py

pub fn resolve_filled_regions(binary: &Mask, skel: &Mask, cfg: &Config) -> Mask {
    let (components, n) = label(binary, true);
    let (h, w) = (binary.h, binary.w);
    let mut area = vec![0i64; n + 1];
    let mut nskel = vec![0i64; n + 1];
    for i in 0..h * w {
        let l = components.data[i] as usize;
        if l != 0 {
            area[l] += 1;
            if skel.data[i] {
                nskel[l] += 1;
            }
        }
    }
    let mut result = skel.clone();
    for cid in 1..=n {
        if area[cid] < cfg.eyes_min_area {
            continue;
        }
        let ns = nskel[cid];
        if ns == 0 || ns > cfg.eyes_max_skeleton_pixels {
            continue;
        }
        if (area[cid] as f64) / (ns as f64) < cfg.eyes_min_fill_ratio {
            continue;
        }
        // region mask
        let mut mask = Mask::new(h, w, false);
        for i in 0..h * w {
            mask.data[i] = components.data[i] as usize == cid;
        }
        let eroded = binary_erosion_3x3(&mask);
        let mut band = Mask::new(h, w, false);
        for i in 0..h * w {
            band.data[i] = mask.data[i] && !eroded.data[i];
        }
        let boundary = skeletonize_zhang(&band);
        for i in 0..h * w {
            if mask.data[i] {
                result.data[i] = false;
            }
        }
        for i in 0..h * w {
            if boundary.data[i] {
                result.data[i] = true;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// crossings.py

fn local_tau(skel: &Mask, dt: &Grid<f64>, radius: usize) -> Grid<f64> {
    let (h, w) = (skel.h, skel.w);
    let window = 2 * radius + 1;
    let mut weighted = Grid::new(h, w, 0.0f64);
    let mut count = Grid::new(h, w, 0.0f64);
    for i in 0..h * w {
        let s = if skel.data[i] { 1.0 } else { 0.0 };
        weighted.data[i] = s * dt.data[i];
        count.data[i] = s;
    }
    let mut wf = Grid::new(h, w, 0.0f64);
    uniform_filter_into(&weighted, window, &mut wf);
    // reuse `weighted` as the count-filter output buffer
    uniform_filter_into(&count, window, &mut weighted);
    let cf = weighted;
    let fallback = if skel.any() {
        let vals: Vec<f64> = (0..h * w).filter(|&i| skel.data[i]).map(|i| dt.data[i]).collect();
        np_median(&vals)
    } else {
        1.0
    };
    let mut out = count; // reuse buffer
    for i in 0..h * w {
        out.data[i] = if cf.data[i] > 0.0 { wf.data[i] / cf.data[i].max(1e-12) } else { fallback };
    }
    out
}

pub struct Crossing {
    pub group_id: usize,
    pub arm_endpoints: Vec<(i64, i64)>,
    pub arm_pairing: Option<((usize, usize), (usize, usize))>,
}

pub struct DetectResult {
    pub group_labels: Grid<i32>,
    pub crossings: Vec<Crossing>,
}

fn arm_tangents_and_endpoints(
    arm_labels: &Grid<i32>,
    n_arms: usize,
    centroid: (f64, f64),
    steps: usize,
) -> (Vec<P2>, Vec<(i64, i64)>) {
    let (cy, cx) = centroid;
    let mut tangents = vec![[0.0, 0.0]; n_arms];
    let mut endpoints = vec![(0i64, 0i64); n_arms];
    // collect pixels per arm in raster order (np.where order)
    let mut pix: Vec<Vec<(i64, i64)>> = vec![Vec::new(); n_arms + 1];
    for y in 0..arm_labels.h {
        for x in 0..arm_labels.w {
            let l = arm_labels.at(y, x);
            if l > 0 {
                pix[l as usize].push((y as i64, x as i64));
            }
        }
    }
    for arm_id in 1..=n_arms {
        let ps = &pix[arm_id];
        let idx = arm_id - 1;
        if ps.is_empty() {
            continue;
        }
        let d2: Vec<f64> = ps.iter().map(|&(y, x)| (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2)).collect();
        let mut order: Vec<usize> = (0..ps.len()).collect();
        order.sort_by(|&a, &b| d2[a].partial_cmp(&d2[b]).unwrap());
        let nhead = 1usize.max(steps.min(order.len()));
        let hy: Vec<f64> = order[..nhead].iter().map(|&i| ps[i].0 as f64).collect();
        let hx: Vec<f64> = order[..nhead].iter().map(|&i| ps[i].1 as f64).collect();
        let head_y = np_sum(&hy) / nhead as f64;
        let head_x = np_sum(&hx) / nhead as f64;
        let vec = [head_y - cy, head_x - cx];
        let n = crate::numeric::norm2(vec);
        if n >= 1e-9 {
            tangents[idx] = [vec[0] / n, vec[1] / n];
        }
        endpoints[idx] = ps[order[0]];
    }
    (tangents, endpoints)
}

fn best_pairing(t: &[P2]) -> (f64, ((usize, usize), (usize, usize))) {
    let matchings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))];
    let mut best_score = f64::NEG_INFINITY;
    let mut best = matchings[0];
    for m in matchings {
        let ((a, b), (c, d)) = m;
        let s = -crate::blas::dot2(t[a], t[b]) - crate::blas::dot2(t[c], t[d]);
        if s > best_score {
            best_score = s;
            best = m;
        }
    }
    (best_score, best)
}

pub fn detect_crossings(binary: &Mask, dt: &Grid<f64>, skel: &Mask, cfg: &Config) -> DetectResult {
    let (h, w) = (binary.h, binary.w);
    let mut timer = crate::StageTimer::new();
    let tau = local_tau(skel, dt, cfg.local_tau_radius);
    timer.lap("    local_tau");
    let mut fat = Mask::new(h, w, false);
    for i in 0..h * w {
        fat.data[i] = binary.data[i] && dt.data[i] > cfg.fat_ratio * tau.data[i];
    }
    let dilated_fat = binary_dilation_cross(&fat, cfg.group_dilate);
    timer.lap("    dilate fat");
    let (group_labels, n_groups) = label(&dilated_fat, true);
    timer.lap("    label groups");
    if crate::timing_enabled() {
        eprintln!("    n_groups={}", n_groups);
    }
    let ring_dilate = cfg.group_dilate + cfg.skel_ring_dilate;
    let mut crossings = Vec::new();
    // bounding boxes per group for locality
    let mut bbox: Vec<(usize, usize, usize, usize)> = vec![(usize::MAX, 0, usize::MAX, 0); n_groups + 1];
    for y in 0..h {
        for x in 0..w {
            let g = group_labels.at(y, x) as usize;
            if g != 0 {
                let b = &mut bbox[g];
                b.0 = b.0.min(y);
                b.1 = b.1.max(y);
                b.2 = b.2.min(x);
                b.3 = b.3.max(x);
            }
        }
    }
    for gid in 1..=n_groups {
        let (gy0, gy1, gx0, gx1) = bbox[gid];
        // fat pixels in group
        let mut fat_pix: Vec<(usize, usize)> = Vec::new();
        for y in gy0..=gy1 {
            for x in gx0..=gx1 {
                if group_labels.at(y, x) as usize == gid && fat.at(y, x) {
                    fat_pix.push((y, x));
                }
            }
        }
        let area = fat_pix.len();
        if area < cfg.min_fat_area {
            continue;
        }
        // arm region: dilate fat_in_group by ring_dilate; operate on a local window
        let pad = ring_dilate + 1;
        let wy0 = gy0.saturating_sub(pad);
        let wx0 = gx0.saturating_sub(pad);
        let wy1 = (gy1 + pad).min(h - 1);
        let wx1 = (gx1 + pad).min(w - 1);
        let lh = wy1 - wy0 + 1;
        let lw = wx1 - wx0 + 1;
        let mut local_fat = Mask::new(lh, lw, false);
        for &(y, x) in &fat_pix {
            local_fat.set(y - wy0, x - wx0, true);
        }
        let arm_region = binary_dilation_cross(&local_fat, ring_dilate);
        let mut arm_skel = Mask::new(lh, lw, false);
        for ly in 0..lh {
            for lx in 0..lw {
                let y = ly + wy0;
                let x = lx + wx0;
                arm_skel.set(ly, lx, arm_region.at(ly, lx) && group_labels.at(y, x) as usize != gid && skel.at(y, x));
            }
        }
        let (arm_labels_l, n_arms) = label(&arm_skel, true);
        // centroid of fat pixels (np.where order: raster)
        let ys: Vec<f64> = fat_pix.iter().map(|p| p.0 as f64).collect();
        let xs: Vec<f64> = fat_pix.iter().map(|p| p.1 as f64).collect();
        let centroid = (np_sum(&ys) / area as f64, np_sum(&xs) / area as f64);
        let chrom_skel_len = fat_pix.iter().filter(|&&(y, x)| skel.at(y, x)).count();
        // Build arm labels grid in global coordinates for tangent computation.
        let mut arm_tangents: Vec<P2> = Vec::new();
        let mut arm_endpoints: Vec<(i64, i64)> = Vec::new();
        if n_arms > 0 {
            // translate local labels to global coordinates by building a Grid with offsets
            let mut glob = Grid::new(lh, lw, 0i32);
            glob.data.copy_from_slice(&arm_labels_l.data);
            let centroid_local = (centroid.0 - wy0 as f64, centroid.1 - wx0 as f64);
            let (t, e) = arm_tangents_and_endpoints(&glob, n_arms, centroid_local, cfg.pairing_tangent_steps);
            arm_tangents = t;
            arm_endpoints = e.into_iter().map(|(y, x)| (y + wy0 as i64, x + wx0 as i64)).collect();
        }
        let mut arm_pairing = None;
        let mut passed = false;
        if n_arms == 4 {
            let (score, pairing) = best_pairing(&arm_tangents);
            arm_pairing = Some(pairing);
            passed = score >= cfg.pairing_threshold;
        }
        if passed && chrom_skel_len >= cfg.min_chromosome_skel_length {
            crossings.push(Crossing { group_id: gid, arm_endpoints, arm_pairing });
        }
    }
    DetectResult { group_labels, crossings }
}

pub fn resolve_crossings(skel: &Mask, det: &DetectResult) -> Mask {
    let mut result = skel.clone();
    let (h, w) = (skel.h, skel.w);
    for c in &det.crossings {
        let pairing = match c.arm_pairing {
            Some(p) => p,
            None => continue,
        };
        for i in 0..h * w {
            if det.group_labels.data[i] as usize == c.group_id {
                result.data[i] = false;
            }
        }
        for (i, j) in [pairing.0, pairing.1] {
            let (y0, x0) = c.arm_endpoints[i];
            let (y1, x1) = c.arm_endpoints[j];
            for (r, cc) in bresenham(y0, x0, y1, x1) {
                if r >= 0 && r < h as i64 && cc >= 0 && cc < w as i64 {
                    result.set(r as usize, cc as usize, true);
                }
            }
        }
    }
    result
}

/// Full skeletonization stage: returns (binary, uncrossed skeleton).
pub fn skeletonize_stage(binary: &Mask, dt: &Grid<f64>, cfg: &Config) -> Mask {
    let mut timer = crate::StageTimer::new();
    let skeletonized = skeletonize_zhang(binary);
    timer.lap("  zhang");
    let collapsed = collapse_small_holes(&skeletonized, cfg);
    timer.lap("  collapse holes (+lee)");
    let eyes_resolved = resolve_filled_regions(binary, &collapsed, cfg);
    timer.lap("  eyes");
    let det = detect_crossings(binary, dt, &eyes_resolved, cfg);
    timer.lap("  detect crossings");
    let r = resolve_crossings(&eyes_resolved, &det);
    timer.lap("  resolve crossings");
    r
}
