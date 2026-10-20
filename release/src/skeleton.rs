//! Ports of `skimage.morphology.skeletonize` (Zhang-Suen and Lee thinning).

use crate::ndimage::Mask;

const ZHANG_LUT: [u8; 256] = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
];

/// Zhang-Suen thinning exactly as `skimage.morphology.skeletonize(method='zhang')`.
pub fn skeletonize_zhang(image: &Mask) -> Mask {
    let nrows = image.h + 2;
    let ncols = image.w + 2;
    let mut skel = vec![0u8; nrows * ncols];
    // ON pixels in raster order; the two sub-passes only ever test ON pixels and
    // read the unmodified previous state, so deferring removals until the end of a
    // sub-pass is exactly the original "skeleton = cleaned_skeleton" copy.
    let mut on: Vec<u32> = Vec::new();
    for y in 0..image.h {
        for x in 0..image.w {
            if image.at(y, x) {
                let i = (y + 1) * ncols + (x + 1);
                skel[i] = 1;
                on.push(i as u32);
            }
        }
    }
    let mut removed: Vec<u32> = Vec::new();
    let mut pixel_removed = true;
    while pixel_removed {
        pixel_removed = false;
        for pass_num in 0..2 {
            let first_pass = pass_num == 0;
            removed.clear();
            for &iu in &on {
                let i = iu as usize;
                if skel[i] == 0 {
                    continue;
                }
                let idx = skel[i - ncols - 1] as usize
                    + 2 * skel[i - ncols] as usize
                    + 4 * skel[i - ncols + 1] as usize
                    + 8 * skel[i + 1] as usize
                    + 16 * skel[i + ncols + 1] as usize
                    + 32 * skel[i + ncols] as usize
                    + 64 * skel[i + ncols - 1] as usize
                    + 128 * skel[i - 1] as usize;
                let neighbors = ZHANG_LUT[idx];
                if neighbors == 0 {
                    continue;
                } else if neighbors == 3 || (neighbors == 1 && first_pass) || (neighbors == 2 && !first_pass) {
                    removed.push(iu);
                }
            }
            if !removed.is_empty() {
                pixel_removed = true;
                for &iu in &removed {
                    skel[iu as usize] = 0;
                }
            }
        }
        on.retain(|&iu| skel[iu as usize] != 0);
    }
    let mut out = Mask::new(image.h, image.w, false);
    for &iu in &on {
        let i = iu as usize;
        let y = i / ncols - 1;
        let x = i % ncols - 1;
        out.set(y, x, true);
    }
    out
}

// ---------------------------------------------------------------------------
// Lee 94 thinning (skimage `_compute_thin_image`), specialised to a 2-D slab
// (3 planes: zero, image, zero).

const EULER_ARR: [i32; 128] = [
    1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3,
    3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1,
    1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1,
    1, 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1,
];

fn euler_lut() -> [i32; 256] {
    let mut lut = [0i32; 256];
    for i in 0..128 {
        lut[2 * i + 1] = EULER_ARR[i];
    }
    lut
}

const NEIGHB_IDX: [[usize; 7]; 8] = [
    [2, 1, 11, 10, 5, 4, 14],
    [0, 9, 3, 12, 1, 10, 4],
    [8, 7, 17, 16, 5, 4, 14],
    [6, 15, 7, 16, 3, 12, 4],
    [20, 23, 19, 22, 11, 14, 10],
    [18, 21, 9, 12, 19, 22, 10],
    [26, 23, 17, 14, 25, 22, 16],
    [24, 25, 15, 16, 21, 22, 12],
];

fn is_euler_invariant(nb: &[u8; 27], lut: &[i32; 256]) -> bool {
    let mut euler_char = 0;
    for oct in 0..8 {
        let mut n: usize = 1;
        for j in 0..7 {
            if nb[NEIGHB_IDX[oct][j]] == 1 {
                n |= 1 << (7 - j);
            }
        }
        euler_char += lut[n];
    }
    euler_char == 0
}

fn is_endpoint(nb: &[u8; 27]) -> bool {
    let s: u32 = nb.iter().map(|&v| v as u32).sum();
    s == 2
}

// octree tables: (indices, list of new octants per index)
const OCTREE: [([usize; 7], [&[usize]; 7]); 8] = [
    ([0, 1, 3, 4, 9, 10, 12], [&[], &[2], &[3], &[2, 3, 4], &[5], &[2, 5, 6], &[3, 5, 7]]),
    ([1, 4, 10, 2, 5, 11, 13], [&[1], &[1, 3, 4], &[1, 5, 6], &[], &[4], &[6], &[4, 6, 8]]),
    ([3, 4, 12, 6, 7, 14, 15], [&[1], &[1, 2, 4], &[1, 5, 7], &[], &[4], &[7], &[4, 7, 8]]),
    ([4, 5, 13, 7, 15, 8, 16], [&[1, 2, 3], &[2], &[2, 6, 8], &[3], &[3, 7, 8], &[], &[8]]),
    ([9, 10, 12, 17, 18, 20, 21], [&[1], &[1, 2, 6], &[1, 3, 7], &[], &[6], &[7], &[6, 7, 8]]),
    ([10, 11, 13, 18, 21, 19, 22], [&[1, 2, 5], &[2], &[2, 4, 8], &[5], &[5, 7, 8], &[], &[8]]),
    ([12, 14, 15, 20, 21, 23, 24], [&[1, 3, 5], &[3], &[3, 4, 8], &[5], &[5, 6, 8], &[], &[8]]),
    ([13, 15, 16, 21, 22, 24, 25], [&[2, 4, 6], &[3, 4, 7], &[4], &[5, 6, 7], &[6], &[7], &[]]),
];

fn octree_labeling(octant: usize, label: u8, cube: &mut [u8; 26]) {
    let (indices, lists) = &OCTREE[octant - 1];
    for k in 0..7 {
        let idx = indices[k];
        if cube[idx] == 1 {
            cube[idx] = label;
            for &no in lists[k].iter() {
                octree_labeling(no, label, cube);
            }
        }
    }
}

fn is_simple_point(nb: &[u8; 27]) -> bool {
    let mut cube = [0u8; 26];
    cube[..13].copy_from_slice(&nb[..13]);
    cube[13..].copy_from_slice(&nb[14..27]);
    let mut label: u8 = 2;
    for i in 0..26 {
        if cube[i] == 1 {
            let oct = match i {
                0 | 1 | 3 | 4 | 9 | 10 | 12 => 1,
                2 | 5 | 11 | 13 => 2,
                6 | 7 | 14 | 15 => 3,
                8 | 16 => 4,
                17 | 18 | 20 | 21 => 5,
                19 | 22 => 6,
                23 | 24 => 7,
                25 => 8,
                _ => unreachable!(),
            };
            octree_labeling(oct, label, &mut cube);
            label += 1;
            if label - 2 >= 2 {
                return false;
            }
        }
    }
    true
}

struct Vol {
    np: usize,
    nr: usize,
    nc: usize,
    data: Vec<u8>,
}
impl Vol {
    #[inline]
    fn at(&self, p: usize, r: usize, c: usize) -> u8 {
        self.data[(p * self.nr + r) * self.nc + c]
    }
    #[inline]
    fn set(&mut self, p: usize, r: usize, c: usize, v: u8) {
        self.data[(p * self.nr + r) * self.nc + c] = v;
    }
    fn neighborhood(&self, p: usize, r: usize, c: usize, nb: &mut [u8; 27]) {
        let mut k = 0;
        for dp in [p - 1, p, p + 1] {
            for dc in [c - 1, c, c + 1] {
                for dr in [r - 1, r, r + 1] {
                    // ordering per get_neighborhood: index = dp*9 + dc*3 + dr
                    nb[k] = self.at(dp, dr, dc);
                    k += 1;
                }
            }
        }
    }
}

/// Lee thinning exactly as `skimage.morphology.skeletonize(method='lee')` for 2-D input.
pub fn skeletonize_lee(image: &Mask) -> Mask {
    let lut = euler_lut();
    let nr = image.h + 2;
    let nc = image.w + 2;
    let np_ = 3;
    let mut vol = Vol { np: np_, nr, nc, data: vec![0u8; np_ * nr * nc] };
    for y in 0..image.h {
        for x in 0..image.w {
            if image.at(y, x) {
                vol.set(1, y + 1, x + 1, 1);
            }
        }
    }
    let borders = [4, 3, 2, 1, 5, 6];
    let num_borders = if vol.np == 3 { 4 } else { 6 };
    let mut unchanged_borders = 0;
    let mut nb = [0u8; 27];
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
    // ON voxels in (p, r, c) scan order (the only ones the candidate scan visits)
    let mut on: Vec<(usize, usize, usize)> = Vec::new();
    for p in 1..vol.np - 1 {
        for r in 1..vol.nr - 1 {
            for c in 1..vol.nc - 1 {
                if vol.at(p, r, c) != 0 {
                    on.push((p, r, c));
                }
            }
        }
    }
    while unchanged_borders < num_borders {
        unchanged_borders = 0;
        for j in 0..num_borders {
            let curr_border = borders[j];
            candidates.clear();
            on.retain(|&(p, r, c)| vol.at(p, r, c) != 0);
            for &(p, r, c) in &on {
                {
                    {
                        let is_border_pt = (curr_border == 1 && vol.at(p, r, c - 1) == 0)
                            || (curr_border == 2 && vol.at(p, r, c + 1) == 0)
                            || (curr_border == 3 && vol.at(p, r + 1, c) == 0)
                            || (curr_border == 4 && vol.at(p, r - 1, c) == 0)
                            || (curr_border == 5 && vol.at(p + 1, r, c) == 0)
                            || (curr_border == 6 && vol.at(p - 1, r, c) == 0);
                        if !is_border_pt {
                            continue;
                        }
                        vol.neighborhood(p, r, c, &mut nb);
                        if is_endpoint(&nb) || !is_euler_invariant(&nb, &lut) || !is_simple_point(&nb) {
                            continue;
                        }
                        candidates.push((p, r, c));
                    }
                }
            }
            let mut no_change = true;
            for &(p, r, c) in &candidates {
                vol.neighborhood(p, r, c, &mut nb);
                if is_simple_point(&nb) {
                    vol.set(p, r, c, 0);
                    no_change = false;
                }
            }
            if no_change {
                unchanged_borders += 1;
            }
        }
    }
    let mut out = Mask::new(image.h, image.w, false);
    for y in 0..image.h {
        for x in 0..image.w {
            out.set(y, x, vol.at(1, y + 1, x + 1) != 0);
        }
    }
    out
}
