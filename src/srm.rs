use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray2};

/// Truncation threshold, and number of bins per dimension.
const T: i32 = 2;
const B: usize = (2 * T + 1) as usize;

/// Residual kernel, residual[i, j] = sum c * x[i + dy, j + dx], of shape [h - crop_y, w - crop_x].
pub(crate) struct Kernel {
    crop: (usize, usize),
    terms: &'static [(usize, usize, i32)],
}

// Generated from sealwatch/srm/cooccurrence.py
const K1_R: Kernel = Kernel { crop: (2, 2), terms: &[(1, 2, 1), (1, 1, -1)] };
const K1_L: Kernel = Kernel { crop: (2, 2), terms: &[(1, 0, 1), (1, 1, -1)] };
const K1_U: Kernel = Kernel { crop: (2, 2), terms: &[(0, 1, 1), (1, 1, -1)] };
const K1_D: Kernel = Kernel { crop: (2, 2), terms: &[(2, 1, 1), (1, 1, -1)] };
const K1_RU: Kernel = Kernel { crop: (2, 2), terms: &[(0, 2, 1), (1, 1, -1)] };
const K1_LU: Kernel = Kernel { crop: (2, 2), terms: &[(0, 0, 1), (1, 1, -1)] };
const K1_RD: Kernel = Kernel { crop: (2, 2), terms: &[(2, 2, 1), (1, 1, -1)] };
const K1_LD: Kernel = Kernel { crop: (2, 2), terms: &[(2, 0, 1), (1, 1, -1)] };
const K2_DH: Kernel = Kernel { crop: (2, 2), terms: &[(1, 0, 1), (1, 1, -2), (1, 2, 1)] };
const K2_DV: Kernel = Kernel { crop: (2, 2), terms: &[(0, 1, 1), (1, 1, -2), (2, 1, 1)] };
const K2_DD: Kernel = Kernel { crop: (2, 2), terms: &[(0, 0, 1), (1, 1, -2), (2, 2, 1)] };
const K2_DM: Kernel = Kernel { crop: (2, 2), terms: &[(0, 2, 1), (1, 1, -2), (2, 0, 1)] };
const K3_R: Kernel = Kernel { crop: (4, 4), terms: &[(2, 4, -1), (2, 3, 3), (2, 2, -3), (2, 1, 1)] };
const K3_L: Kernel = Kernel { crop: (4, 4), terms: &[(2, 0, -1), (2, 1, 3), (2, 2, -3), (2, 3, 1)] };
const K3_U: Kernel = Kernel { crop: (4, 4), terms: &[(0, 2, -1), (1, 2, 3), (2, 2, -3), (3, 2, 1)] };
const K3_D: Kernel = Kernel { crop: (4, 4), terms: &[(4, 2, -1), (3, 2, 3), (2, 2, -3), (1, 2, 1)] };
const K3_RU: Kernel = Kernel { crop: (4, 4), terms: &[(0, 4, -1), (1, 3, 3), (2, 2, -3), (3, 1, 1)] };
const K3_LU: Kernel = Kernel { crop: (4, 4), terms: &[(0, 0, -1), (1, 1, 3), (2, 2, -3), (3, 3, 1)] };
const K3_RD: Kernel = Kernel { crop: (4, 4), terms: &[(4, 4, -1), (3, 3, 3), (2, 2, -3), (1, 1, 1)] };
const K3_LD: Kernel = Kernel { crop: (4, 4), terms: &[(4, 0, -1), (3, 1, 3), (2, 2, -3), (1, 3, 1)] };
const K3X3_D: Kernel = Kernel { crop: (2, 2), terms: &[(0, 0, -1), (1, 0, 2), (2, 0, -1), (0, 1, 2), (1, 1, -4), (2, 1, 2), (0, 2, -1), (1, 2, 2), (2, 2, -1)] };
const K3X3_DU: Kernel = Kernel { crop: (2, 2), terms: &[(0, 1, 2), (1, 0, 2), (1, 2, 2), (0, 0, -1), (0, 2, -1), (1, 1, -4)] };
const K3X3_DB: Kernel = Kernel { crop: (2, 2), terms: &[(2, 1, 2), (1, 0, 2), (1, 2, 2), (2, 0, -1), (2, 2, -1), (1, 1, -4)] };
const K3X3_DL: Kernel = Kernel { crop: (2, 2), terms: &[(1, 0, 2), (0, 1, 2), (2, 1, 2), (0, 0, -1), (2, 0, -1), (1, 1, -4)] };
const K3X3_DR: Kernel = Kernel { crop: (2, 2), terms: &[(1, 2, 2), (0, 1, 2), (2, 1, 2), (0, 2, -1), (2, 2, -1), (1, 1, -4)] };
const K5X5_D: Kernel = Kernel { crop: (4, 4), terms: &[(1, 2, 8), (3, 2, 8), (2, 1, 8), (2, 3, 8), (1, 3, -6), (1, 1, -6), (3, 1, -6), (3, 3, -6), (0, 2, -2), (4, 2, -2), (2, 4, -2), (2, 0, -2), (1, 0, 2), (0, 1, 2), (0, 3, 2), (1, 4, 2), (3, 4, 2), (4, 3, 2), (4, 1, 2), (3, 0, 2), (0, 0, -1), (0, 4, -1), (4, 0, -1), (4, 4, -1), (2, 2, -12)] };
const K5X5_DU: Kernel = Kernel { crop: (4, 4), terms: &[(2, 1, 8), (1, 2, 8), (2, 3, 8), (1, 1, -6), (1, 3, -6), (2, 0, -2), (2, 4, -2), (0, 2, -2), (1, 0, 2), (0, 1, 2), (0, 3, 2), (1, 4, 2), (0, 0, -1), (0, 4, -1), (2, 2, -12)] };
const K5X5_DR: Kernel = Kernel { crop: (4, 4), terms: &[(1, 2, 8), (2, 3, 8), (3, 2, 8), (1, 3, -6), (3, 3, -6), (0, 2, -2), (4, 2, -2), (2, 4, -2), (0, 3, 2), (1, 4, 2), (3, 4, 2), (4, 3, 2), (0, 4, -1), (4, 4, -1), (2, 2, -12)] };
const K5X5_DB: Kernel = Kernel { crop: (4, 4), terms: &[(2, 3, 8), (3, 2, 8), (2, 1, 8), (3, 3, -6), (3, 1, -6), (2, 0, -2), (2, 4, -2), (4, 2, -2), (3, 4, 2), (4, 3, 2), (4, 1, 2), (3, 0, 2), (4, 4, -1), (4, 0, -1), (2, 2, -12)] };
const K5X5_DL: Kernel = Kernel { crop: (4, 4), terms: &[(3, 2, 8), (2, 1, 8), (1, 2, 8), (3, 1, -6), (1, 1, -6), (0, 2, -2), (4, 2, -2), (2, 0, -2), (4, 1, 2), (3, 0, 2), (1, 0, 2), (0, 1, 2), (4, 0, -1), (0, 0, -1), (2, 2, -12)] };

/// Quantized residual, row-major, as bins in [0, 2T] of the values in [-T, T].
/// Channels of a pixel are stored next to each other.
pub(crate) struct Plane {
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) c: usize,
    pub(crate) v: Vec<u8>,
}

impl Plane {
    fn row(&self, i: usize) -> &[u8] {
        &self.v[i * self.w..(i + 1) * self.w]
    }
}

/// Rounding as in Matlab, as sealwatch.tools.matlab.round, |x| >= 0.5 rounds away from zero.
pub(crate) fn matlab_round(v: f64) -> f64 {
    let a = v.abs();
    let floor = a.floor();
    let frac = a - floor;
    // np.isclose(frac, 0.5)
    let half = frac >= 0.5 || (frac - 0.5).abs() <= 1e-8 + 1e-5 * 0.5;
    let rounded = if half { a.ceil() } else { floor };
    if v < 0.0 { -rounded } else { rounded }
}

/// Residual of the image by a kernel, row-major, of shape [h - crop_y, w - crop_x, c].
/// For 8-bit images, the residuals of all kernels fit 16 bits, |r| <= 48 * 255 = 12240 (KV kernel).
pub(crate) struct Residual {
    w: usize,
    c: usize,
    r: Vec<i16>,
    // range of possible values, for 8-bit images
    low: i32,
    high: i32,
}

/// Residual of an image of shape [h, w, c], with channels of a pixel next to each other.
pub(crate) fn residual(x: &[u8], h: usize, w: usize, c: usize, k: &Kernel) -> Residual {
    let (rh, rw) = (h - k.crop.0, w - k.crop.1);
    let mut r = vec![0i16; rh * rw * c];
    // each term adds a shifted image, each channel shifted by the same pixels
    for (i, row) in r.chunks_exact_mut(rw * c).enumerate() {
        for &(dy, dx, coef) in k.terms {
            let shifted = &x[((i + dy) * w + dx) * c..][..rw * c];
            for (o, &p) in row.iter_mut().zip(shifted) {
                *o += coef as i16 * p as i16;
            }
        }
    }
    let low = k.terms.iter().map(|&(_, _, coef)| 255 * coef.min(0)).sum();
    let high = k.terms.iter().map(|&(_, _, coef)| 255 * coef.max(0)).sum();
    Residual { w: rw, c, r, low, high }
}

/// Residual quantized by round(r), truncated to [-t, t], as bins.
pub(crate) fn quantize(res: &Residual, t: i32, round: impl Fn(i32) -> i32) -> Plane {
    // lookup table over the range of the residual
    let lut: Vec<u8> = (res.low..=res.high).map(|r| (round(r).clamp(-t, t) + t) as u8).collect();
    let v = res.r.iter().map(|&r| lut[(r as i32 - res.low) as usize]).collect();
    Plane { h: res.r.len() / (res.w * res.c), w: res.w, c: res.c, v }
}

fn elementwise(a: &Plane, b: &Plane, f: fn(u8, u8) -> u8) -> Plane {
    Plane { h: a.h, w: a.w, c: a.c, v: a.v.iter().zip(&b.v).map(|(&p, &q)| f(p, q)).collect() }
}

pub(crate) fn min(a: &Plane, b: &Plane) -> Plane {
    elementwise(a, b, std::cmp::min)
}

pub(crate) fn max(a: &Plane, b: &Plane) -> Plane {
    elementwise(a, b, std::cmp::max)
}

#[derive(Clone, Copy)]
pub(crate) enum Dir {
    Hor,
    Ver,
}

/// Normalized co-occurrence of the planes, in a direction.
pub(crate) type Cooc<'a> = dyn Fn(&[&Plane], Dir) -> Vec<f64> + 'a;

/// Counts of 4th-order co-occurrences of bins in [0, B).
///
/// Counted in four copies, window k into copy k mod 4,
/// so that consecutive windows falling into the same bin do not wait for each other.
struct Counts {
    copies: [Box<[u32; SLOTS]>; 4],
}

/// Slots of a copy, enough for the index of any pair of u8, so that the compiler can omit bounds checks.
/// Only the first B^4 are used.
const SLOTS: usize = 256 * B * B + 256;

impl Counts {
    fn new() -> Self {
        Counts { copies: std::array::from_fn(|_| Box::new([0; SLOTS])) }
    }

    /// Counts the windows (first[k], second[k]) of pairs of bins, from pairs().
    fn add(&mut self, first: &[u8], second: &[u8]) {
        let n = first.len();
        let second = &second[..n];
        let index = |k: usize| first[k] as usize * B * B + second[k] as usize;
        // four windows per step, written out, so that they are counted independently
        let [c0, c1, c2, c3] = &mut self.copies;
        let mut k = 0;
        while k + 4 <= n {
            c0[index(k)] += 1;
            c1[index(k + 1)] += 1;
            c2[index(k + 2)] += 1;
            c3[index(k + 3)] += 1;
            k += 4;
        }
        for k in k..n {
            c0[index(k)] += 1;
        }
    }

    /// Counts normalized to unit sum.
    fn normalized(&self) -> Vec<f64> {
        let counts: Vec<u64> = (0..B * B * B * B).map(|i| self.copies.iter().map(|c| c[i] as u64).sum()).collect();
        let total: u64 = counts.iter().sum();
        counts.iter().map(|&c| c as f64 / total as f64).collect()
    }
}

/// Pairs of consecutive bins (a[k], b[k]), as a single bin in [0, B^2).
fn pairs(out: &mut [u8], a: &[u8], b: &[u8]) {
    for ((o, &p), &q) in out.iter_mut().zip(a).zip(b) {
        *o = p * B as u8 + q;
    }
}

/// Normalized 4th-order co-occurrence of horizontally or vertically neighboring values,
/// over all planes, as of the vertically or horizontally stacked planes.
///
/// A window of four bins consists of two pairs, the pairs at k and k + 2.
fn cooc4(planes: &[&Plane], dir: Dir) -> Vec<f64> {
    let mut counts = Counts::new();
    // pairs, reused across rows and planes, as cooc() is called hundreds of times per image
    let mut pr = Vec::new();
    for p in planes {
        match dir {
            Dir::Hor if p.w >= 4 => {
                // pairs of a row
                pr.resize(p.w - 1, 0);
                for i in 0..p.h {
                    let r = p.row(i);
                    pairs(&mut pr, &r[..p.w - 1], &r[1..]);
                    counts.add(&pr[..p.w - 3], &pr[2..]);
                }
            }
            Dir::Ver if p.h >= 4 => {
                // pairs of all consecutive rows
                pr.resize((p.h - 1) * p.w, 0);
                for (i, out) in pr.chunks_exact_mut(p.w).enumerate() {
                    pairs(out, p.row(i), p.row(i + 1));
                }
                for i in 0..p.h - 3 {
                    counts.add(&pr[i * p.w..(i + 1) * p.w], &pr[(i + 2) * p.w..(i + 3) * p.w]);
                }
            }
            _ => {}
        }
    }
    counts.normalized()
}

/// Sum of two co-occurrences.
pub(crate) fn add(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    a.iter().zip(&b).map(|(p, q)| p + q).collect()
}

/// Co-occurrences of a family, by submodel name.
pub(crate) type Submodels = Vec<(&'static str, Vec<f64>)>;

/// Co-occurrences of all 1st-order residuals, quantized with residuals of FAMILY_KERNELS,
/// with co-occurrence cooc, as all1st() in sealwatch/srm/cooccurrence.py.
pub(crate) fn all1st(y: &[Plane], cooc: &Cooc, directional: bool) -> Submodels {
    use Dir::*;
    let [r, l, u, d, ru, lu, rd, ld] = y else { unreachable!() };
    // a horizontally, b vertically
    let hv = |a: &[&Plane], b: &[&Plane]| add(cooc(a, Hor), cooc(b, Ver));
    // a horizontally, and vertically if directional
    let h_or_hv = |a: &[&Plane]| if directional { hv(a, a) } else { cooc(a, Hor) };
    let mut g = Submodels::new();

    // minmax22
    let (rl_min, ud_min, rl_max, ud_max) = (min(r, l), min(u, d), max(r, l), max(u, d));
    g.push(("min22h", hv(&[&rl_min], &[&ud_min])));
    g.push(("max22h", hv(&[&rl_max], &[&ud_max])));
    if directional {
        g.push(("min22v", hv(&[&ud_min], &[&rl_min])));
        g.push(("max22v", hv(&[&ud_max], &[&rl_max])));
    }

    // spam14h/v
    g.push(("spam14h", hv(&[r], &[u])));
    if directional {
        g.push(("spam14v", hv(&[u], &[r])));
    }

    // minmax24
    let (ru_min, rd_min, lu_min, ld_min) = (min(r, u), min(r, d), min(l, u), min(l, d));
    let (ru_max, rd_max, lu_max, ld_max) = (max(r, u), max(r, d), max(l, u), max(l, d));
    let mins = [&ru_min, &rd_min, &lu_min, &ld_min];
    let maxs = [&ru_max, &rd_max, &lu_max, &ld_max];
    g.push(("min24", hv(&mins, &mins)));
    g.push(("max24", hv(&maxs, &maxs)));

    // minmax34h/v
    let (u_min, r_min, d_min, l_min) = (min(&min(l, u), r), min(&min(u, r), d), min(&min(r, d), l), min(&min(d, l), u));
    let (u_max, r_max, d_max, l_max) = (max(&max(l, u), r), max(&max(u, r), d), max(&max(r, d), l), max(&max(d, l), u));
    g.push(("min34h", hv(&[&u_min, &d_min], &[&l_min, &r_min])));
    g.push(("max34h", hv(&[&u_max, &d_max], &[&r_max, &l_max])));
    if directional {
        g.push(("min34v", hv(&[&r_min, &l_min], &[&u_min, &d_min])));
        g.push(("max34v", hv(&[&r_max, &l_max], &[&u_max, &d_max])));
    }

    // minmax41
    let (all_min, all_max) = (min(&rl_min, &ud_min), max(&rl_max, &ud_max));
    g.push(("min41", h_or_hv(&[&all_min])));
    g.push(("max41", h_or_hv(&[&all_max])));

    // minmax34
    let (ru_min, rd_min, lu_min, ld_min) = (min(&ru_min, ru), min(&rd_min, rd), min(&lu_min, lu), min(&ld_min, ld));
    let (ru_max, rd_max, lu_max, ld_max) = (max(&ru_max, ru), max(&rd_max, rd), max(&lu_max, lu), max(&ld_max, ld));
    let mins = [&ru_min, &rd_min, &lu_min, &ld_min];
    let maxs = [&ru_max, &rd_max, &lu_max, &ld_max];
    g.push(("min34", hv(&mins, &mins)));
    g.push(("max34", hv(&maxs, &maxs)));

    // minmax48h/v
    let (ru_min2, rd_min2, ld_min2, lu_min2) = (min(&ru_min, lu), min(&rd_min, ru), min(&ld_min, rd), min(&lu_min, ld));
    let (ru_min3, rd_min3, ld_min3, lu_min3) = (min(&ru_min, rd), min(&rd_min, ld), min(&ld_min, lu), min(&lu_min, ru));
    let (a, b) = ([&ru_min2, &ld_min2, &rd_min3, &lu_min3], [&rd_min2, &lu_min2, &ru_min3, &ld_min3]);
    g.push(("min48h", hv(&a, &b)));
    if directional {
        g.push(("min48v", hv(&b, &a)));
    }
    let (ru_max2, rd_max2, ld_max2, lu_max2) = (max(&ru_max, lu), max(&rd_max, ru), max(&ld_max, rd), max(&lu_max, ld));
    let (ru_max3, rd_max3, ld_max3, lu_max3) = (max(&ru_max, rd), max(&rd_max, ld), max(&ld_max, lu), max(&lu_max, ru));
    let (a, b) = ([&ru_max2, &ld_max2, &rd_max3, &lu_max3], [&rd_max2, &lu_max2, &ru_max3, &ld_max3]);
    g.push(("max48h", hv(&a, &b)));
    if directional {
        g.push(("max48v", hv(&b, &a)));
    }

    // minmax54
    let (ru_min4, rd_min4, ld_min4, lu_min4) = (min(&ru_min2, rd), min(&rd_min2, ld), min(&ld_min2, lu), min(&lu_min2, ru));
    let (ru_min5, rd_min5, ld_min5, lu_min5) = (min(&ru_min3, lu), min(&rd_min3, ru), min(&ld_min3, rd), min(&lu_min3, ld));
    g.push(("min54", hv(&[&ru_min4, &ld_min4, &rd_min5, &lu_min5], &[&rd_min4, &lu_min4, &ru_min5, &ld_min5])));
    let (ru_max4, rd_max4, ld_max4, lu_max4) = (max(&ru_max2, rd), max(&rd_max2, ld), max(&ld_max2, lu), max(&lu_max2, ru));
    let (ru_max5, rd_max5, ld_max5, lu_max5) = (max(&ru_max3, lu), max(&rd_max3, ru), max(&ld_max3, rd), max(&lu_max3, ld));
    g.push(("max54", hv(&[&ru_max4, &ld_max4, &rd_max5, &lu_max5], &[&rd_max4, &lu_max4, &ru_max5, &ld_max5])));
    g
}

/// Co-occurrences of all 2nd-order residuals, quantized with residuals of FAMILY_KERNELS,
/// with co-occurrence cooc, as all2nd() in sealwatch/srm/cooccurrence.py.
pub(crate) fn all2nd(y: &[Plane], cooc: &Cooc, directional: bool) -> Submodels {
    use Dir::*;
    let [yh, yv, yd, ym] = y else { unreachable!() };
    // a horizontally, b vertically
    let hv = |a: &[&Plane], b: &[&Plane]| add(cooc(a, Hor), cooc(b, Ver));
    // a horizontally, and vertically if directional
    let h_or_hv = |a: &[&Plane]| if directional { hv(a, a) } else { cooc(a, Hor) };
    let mut g = Submodels::new();

    // spam12h/v
    g.push(("spam12h", hv(&[yh], &[yv])));
    if directional {
        g.push(("spam12v", hv(&[yv], &[yh])));
    }

    // minmax21
    let (dmin, dmax) = (min(yh, yv), max(yh, yv));
    g.push(("min21", h_or_hv(&[&dmin])));
    g.push(("max21", h_or_hv(&[&dmax])));

    // minmax41
    let (dmin2, dmax2) = (min(&min(&dmin, yd), ym), max(&max(&dmax, yd), ym));
    g.push(("min41", h_or_hv(&[&dmin2])));
    g.push(("max41", h_or_hv(&[&dmax2])));

    // minmax32
    let (ru_min, rd_min, ru_max, rd_max) = (min(&dmin, ym), min(&dmin, yd), max(&dmax, ym), max(&dmax, yd));
    g.push(("min32", h_or_hv(&[&ru_min, &rd_min])));
    g.push(("max32", h_or_hv(&[&ru_max, &rd_max])));

    // minmax24h/v
    let (ru_min2, rd_min2, ru_min3, lu_min3) = (min(ym, yh), min(yd, yh), min(ym, yv), min(yd, yv));
    g.push(("min24h", hv(&[&ru_min2, &rd_min2], &[&ru_min3, &lu_min3])));
    if directional {
        g.push(("min24v", hv(&[&ru_min3, &lu_min3], &[&ru_min2, &rd_min2])));
    }
    let (ru_max2, rd_max2, ru_max3, lu_max3) = (max(ym, yh), max(yd, yh), max(ym, yv), max(yd, yv));
    g.push(("max24h", hv(&[&ru_max2, &rd_max2], &[&ru_max3, &lu_max3])));
    if directional {
        g.push(("max24v", hv(&[&ru_max3, &lu_max3], &[&ru_max2, &rd_max2])));
    }
    g
}

/// Co-occurrences of all 3rd-order residuals, quantized with residuals of FAMILY_KERNELS,
/// with co-occurrence cooc, as all3rd() in sealwatch/srm/cooccurrence.py.
pub(crate) fn all3rd(y: &[Plane], cooc: &Cooc, directional: bool) -> Submodels {
    use Dir::*;
    let [r, l, u, d, ru, lu, rd, ld] = y else { unreachable!() };
    // a horizontally, b vertically
    let hv = |a: &[&Plane], b: &[&Plane]| add(cooc(a, Hor), cooc(b, Ver));
    // a horizontally, and vertically if directional
    let h_or_hv = |a: &[&Plane]| if directional { hv(a, a) } else { cooc(a, Hor) };
    let mut g = Submodels::new();

    // minmax22
    let (rl_min, ud_min, rl_max, ud_max) = (min(r, l), min(u, d), max(r, l), max(u, d));
    g.push(("min22h", hv(&[&rl_min], &[&ud_min])));
    g.push(("max22h", hv(&[&rl_max], &[&ud_max])));
    if directional {
        g.push(("min22v", hv(&[&ud_min], &[&rl_min])));
        g.push(("max22v", hv(&[&ud_max], &[&rl_max])));
    }

    // minmax34h/v
    let (u_min, r_min, d_min, l_min) = (min(&rl_min, u), min(&ud_min, r), min(&rl_min, d), min(&ud_min, l));
    let (u_max, r_max, d_max, l_max) = (max(&rl_max, u), max(&ud_max, r), max(&rl_max, d), max(&ud_max, l));
    g.push(("min34h", hv(&[&u_min, &d_min], &[&r_min, &l_min])));
    g.push(("max34h", hv(&[&u_max, &d_max], &[&r_max, &l_max])));
    if directional {
        g.push(("min34v", hv(&[&r_min, &l_min], &[&u_min, &d_min])));
        g.push(("max34v", hv(&[&r_max, &l_max], &[&u_max, &d_max])));
    }

    // spam14h/v
    g.push(("spam14h", hv(&[r], &[u])));
    if directional {
        g.push(("spam14v", hv(&[u], &[r])));
    }

    // minmax24, as CRM_FIX_MIN24, the vertical part only if directional, like the other submodels
    let (ru_min, rd_min, lu_min, ld_min) = (min(r, u), min(r, d), min(l, u), min(l, d));
    let (ru_max, rd_max, lu_max, ld_max) = (max(r, u), max(r, d), max(l, u), max(l, d));
    let mins = [&ru_min, &rd_min, &lu_min, &ld_min];
    let maxs = [&ru_max, &rd_max, &lu_max, &ld_max];
    g.push(("min24", h_or_hv(&mins)));
    g.push(("max24", h_or_hv(&maxs)));

    // minmax41
    let (all_min, all_max) = (min(&ru_min, &ld_min), max(&ru_max, &ld_max));
    g.push(("min41", h_or_hv(&[&all_min])));
    g.push(("max41", h_or_hv(&[&all_max])));

    // minmax34
    let (ru_min2, rd_min2, lu_min2, ld_min2) = (min(&ru_min, ru), min(&rd_min, rd), min(&lu_min, lu), min(&ld_min, ld));
    let (ru_max2, rd_max2, lu_max2, ld_max2) = (max(&ru_max, ru), max(&rd_max, rd), max(&lu_max, lu), max(&ld_max, ld));
    g.push(("min34", h_or_hv(&[&ru_min2, &rd_min2, &lu_min2, &ld_min2])));
    g.push(("max34", h_or_hv(&[&ru_max2, &rd_max2, &lu_max2, &ld_max2])));

    // minmax48h/v
    let (ru_min3, rd_min3, ld_min3, lu_min3) = (min(&ru_min2, lu), min(&rd_min2, ru), min(&ld_min2, rd), min(&lu_min2, ld));
    let (ru_min4, rd_min4, ld_min4, lu_min4) = (min(&ru_min2, rd), min(&rd_min2, ld), min(&ld_min2, lu), min(&lu_min2, ru));
    let (a, b) = ([&ru_min3, &ld_min3, &rd_min4, &lu_min4], [&rd_min3, &lu_min3, &ru_min4, &ld_min4]);
    g.push(("min48h", hv(&a, &b)));
    if directional {
        g.push(("min48v", hv(&b, &a)));
    }
    let (ru_max3, rd_max3, ld_max3, lu_max3) = (max(&ru_max2, lu), max(&rd_max2, ru), max(&ld_max2, rd), max(&lu_max2, ld));
    let (ru_max4, rd_max4, ld_max4, lu_max4) = (max(&ru_max2, rd), max(&rd_max2, ld), max(&ld_max2, lu), max(&lu_max2, ru));
    let (a, b) = ([&ru_max3, &ld_max3, &rd_max4, &lu_max4], [&rd_max3, &lu_max3, &ru_max4, &ld_max4]);
    g.push(("max48h", hv(&a, &b)));
    if directional {
        g.push(("max48v", hv(&b, &a)));
    }

    // minmax54
    let (ru_min5, rd_min5, ld_min5, lu_min5) = (min(&ru_min3, rd), min(&rd_min3, ld), min(&ld_min3, lu), min(&lu_min3, ru));
    let (ru_max5, rd_max5, ld_max5, lu_max5) = (max(&ru_max3, rd), max(&rd_max3, ld), max(&ld_max3, lu), max(&lu_max3, ru));
    g.push(("min54", h_or_hv(&[&ru_min5, &ld_min5, &rd_min5, &lu_min5])));
    g.push(("max54", h_or_hv(&[&ru_max5, &ld_max5, &rd_max5, &lu_max5])));
    g
}

/// Co-occurrences of the KB (3x3) or KV (5x5) residual and its halves (EDGE residuals),
/// quantized with residuals of FAMILY_KERNELS, with co-occurrence cooc,
/// as all3x3() and all5x5() in sealwatch/srm/cooccurrence.py.
pub(crate) fn all_edge(y: &[Plane], cooc: &Cooc, directional: bool) -> Submodels {
    use Dir::*;
    let [y, yu, yb, yl, yr] = y else { unreachable!() };
    // a horizontally, b vertically
    let hv = |a: &[&Plane], b: &[&Plane]| add(cooc(a, Hor), cooc(b, Ver));
    // a horizontally, and vertically if directional
    let h_or_hv = |a: &[&Plane]| if directional { hv(a, a) } else { cooc(a, Hor) };
    let mut g = Submodels::new();

    // spam11
    g.push(("spam11", h_or_hv(&[y])));

    // spam14h/v
    g.push(("spam14h", hv(&[yu, yb], &[yl, yr])));
    if directional {
        g.push(("spam14v", hv(&[yl, yr], &[yu, yb])));
    }

    // minmax24
    let (dmin1, dmin2, dmin3, dmin4) = (min(yu, yl), min(yb, yr), min(yu, yr), min(yb, yl));
    let (dmax1, dmax2, dmax3, dmax4) = (max(yu, yl), max(yb, yr), max(yu, yr), max(yb, yl));
    g.push(("min24", h_or_hv(&[&dmin1, &dmin2, &dmin3, &dmin4])));
    g.push(("max24", h_or_hv(&[&dmax1, &dmax2, &dmax3, &dmax4])));

    // minmax22
    let (ue_min, re_min, ue_max, re_max) = (min(yu, yb), min(yr, yl), max(yu, yb), max(yr, yl));
    g.push(("min22h", hv(&[&ue_min], &[&re_min])));
    g.push(("max22h", hv(&[&ue_max], &[&re_max])));
    if directional {
        g.push(("min22v", hv(&[&re_min], &[&ue_min])));
        g.push(("max22v", hv(&[&re_max], &[&ue_max])));
    }

    // minmax41
    let (dmin5, dmax5) = (min(&dmin1, &dmin2), max(&dmax1, &dmax2));
    g.push(("min41", h_or_hv(&[&dmin5])));
    g.push(("max41", h_or_hv(&[&dmax5])));
    g
}

/// Index of bin (i, j, k, n), each in [0, B), of a 4D co-occurrence.
fn idx(i: usize, j: usize, k: usize, n: usize) -> usize {
    ((i * B + j) * B + k) * B + n
}

/// Sign and directional symmetrization of a spam co-occurrence, as symm4.
fn symm4(a: &[f32]) -> Vec<f32> {
    let t = T as usize;
    let m = B - 1;
    let mut done = vec![false; a.len()];
    let mut out = vec![0f32; B * B + 4 * t * t * (t + 1) * (t + 1)];
    out[0] = a[idx(t, t, t, t)];
    let mut next = 1;
    for i in 0..B {
        for j in 0..B {
            for k in 0..B {
                for n in 0..B {
                    if done[idx(i, j, k, n)] || (i, j, k, n) == (t, t, t, t) {
                        continue;
                    }
                    out[next] = a[idx(i, j, k, n)] + a[idx(m - i, m - j, m - k, m - n)];
                    done[idx(i, j, k, n)] = true;
                    done[idx(m - i, m - j, m - k, m - n)] = true;
                    if (i != n || j != k) && !done[idx(n, k, j, i)] {
                        out[next] = out[next] + a[idx(n, k, j, i)] + a[idx(m - n, m - k, m - j, m - i)];
                        done[idx(n, k, j, i)] = true;
                        done[idx(m - n, m - k, m - j, m - i)] = true;
                    }
                    next += 1;
                }
            }
        }
    }
    out
}

/// Sign and directional symmetrization of a min and max co-occurrence, as symm4_dir.
fn symm4_dir(fmin: &[f32], fmax: &[f32]) -> Vec<f32> {
    let m = B - 1;
    let t = T as usize;
    // sign symmetrization, min with flipped max
    let mut a = vec![0f32; fmin.len()];
    for i in 0..B {
        for j in 0..B {
            for k in 0..B {
                for n in 0..B {
                    a[idx(i, j, k, n)] = fmin[idx(i, j, k, n)] + fmax[idx(m - i, m - j, m - k, m - n)];
                }
            }
        }
    }
    // directional symmetrization
    let mut done = vec![false; a.len()];
    let mut out = Vec::with_capacity(B * B * B * B - 2 * t * (t + 1) * B * B);
    for i in 0..B {
        for j in 0..B {
            for k in 0..B {
                for n in 0..B {
                    if i != n || j != k {
                        if !done[idx(i, j, k, n)] {
                            out.push(a[idx(i, j, k, n)] + a[idx(n, k, j, i)]);
                            done[idx(i, j, k, n)] = true;
                            done[idx(n, k, j, i)] = true;
                        }
                    } else {
                        out.push(a[idx(i, j, k, n)]);
                        done[idx(i, j, k, n)] = true;
                    }
                }
            }
        }
    }
    out
}

/// Residual kernels of each family, in the order expected by its function.
pub(crate) const FAMILY_KERNELS: [&[&Kernel]; 5] = [
    &[&K1_R, &K1_L, &K1_U, &K1_D, &K1_RU, &K1_LU, &K1_RD, &K1_LD],
    &[&K2_DH, &K2_DV, &K2_DD, &K2_DM],
    &[&K3_R, &K3_L, &K3_U, &K3_D, &K3_RU, &K3_LU, &K3_RD, &K3_LD],
    &[&K3X3_D, &K3X3_DU, &K3X3_DB, &K3X3_DL, &K3X3_DR],
    &[&K5X5_D, &K5X5_DU, &K5X5_DB, &K5X5_DL, &K5X5_DR],
];

/// Order of the submodels of each family, as in the original Matlab implementation,
/// see sealwatch/srm/srm.py
pub(crate) const FAMILIES: [&str; 5] = ["s1", "s2", "s3", "s3x3", "s5x5"];
const SUBMODEL_ORDER: [&[&str]; 5] = [
    &["minmax22h", "minmax34h", "minmax22v", "minmax24", "minmax34v", "minmax41", "minmax34", "minmax48h", "minmax48v", "minmax54", "spam14hv"],
    &["minmax21", "minmax41", "minmax32", "minmax24h", "minmax24v", "spam12hv"],
    &["minmax22h", "minmax34h", "minmax22v", "minmax24", "minmax34v", "minmax41", "minmax34", "minmax48h", "minmax48v", "minmax54", "spam14hv"],
    &["minmax24", "minmax22h", "minmax22v", "minmax41", "spam11", "spam14hv"],
    &["minmax24", "minmax22h", "minmax22v", "minmax41", "spam11", "spam14hv"],
];

/// Symmetrized submodels of a family, by name.
fn post_process(g: Submodels, family: &str, q: &str, out: &mut HashMap<String, Vec<f32>>) {
    let g: HashMap<&str, Vec<f32>> = g.into_iter()
        .map(|(name, f)| (name, f.iter().map(|&v| v as f32).collect()))
        .collect();
    for (name, f) in &g {
        if let Some(suffix) = name.strip_prefix("min") {
            let fmax = &g[format!("max{suffix}").as_str()];
            out.insert(format!("{family}_minmax{suffix}_{q}"), symm4_dir(f, fmax));
        } else if name.starts_with("spam") {
            out.insert(format!("{family}_{name}_{q}"), symm4(f));
        }
    }
    // spam h and v united
    for spam in ["spam12", "spam14"] {
        let (kh, kv) = (format!("{family}_{spam}h_{q}"), format!("{family}_{spam}v_{q}"));
        if let (Some(fh), Some(fv)) = (out.remove(&kh), out.remove(&kv)) {
            out.insert(format!("{family}_{spam}hv_{q}"), [fh, fv].concat());
        }
    }
}

/// Spatial rich model features, in the order of the original Matlab implementation.
fn compute(x: &[u8], h: usize, w: usize, qs: &[Vec<f64>], q_names: &[Vec<String>]) -> Vec<(String, Vec<f32>)> {
    let mut out = HashMap::new();
    let multipliers = [1.0, 2.0, 3.0, 4.0, 12.0];
    for f in 0..qs.len() {
        if qs[f].is_empty() {
            continue;
        }
        // residuals, independent of the quantization step
        let residuals: Vec<Residual> = FAMILY_KERNELS[f].iter().map(|k| residual(x, h, w, 1, k)).collect();
        for (q, q_name) in qs[f].iter().zip(&q_names[f]) {
            let qc = q * multipliers[f];
            let y: Vec<Plane> = residuals.iter()
                .map(|r| quantize(r, T, |r| matlab_round(r as f64 / qc) as i32))
                .collect();
            let g = match f {
                0 => all1st(&y, &cooc4, true),
                1 => all2nd(&y, &cooc4, true),
                2 => all3rd(&y, &cooc4, true),
                _ => all_edge(&y, &cooc4, true),
            };
            post_process(g, FAMILIES[f], &format!("q{q_name}"), &mut out);
        }
    }
    // spam11 of 3x3 (KB) and 5x5 (KV) united
    let names: Vec<String> = out.keys().filter(|k| k.starts_with("s3x3_spam11_")).cloned().collect();
    for k3 in names {
        let k5 = k3.replacen("s3x3", "s5x5", 1);
        if out.contains_key(&k5) {
            let (f3, f5) = (out.remove(&k3).unwrap(), out.remove(&k5).unwrap());
            out.insert(k3.replacen("s3x3", "s35", 1), [f3, f5].concat());
        }
    }
    // order as in Matlab
    let mut ordered = Vec::new();
    for f in 0..qs.len() {
        let mut seen = Vec::new();
        for q_name in &q_names[f] {
            if seen.contains(&q_name) {
                continue;
            }
            seen.push(q_name);
            for submodel in SUBMODEL_ORDER[f] {
                let mut candidates = vec![format!("{}_{submodel}_q{q_name}", FAMILIES[f])];
                if f == 4 && *submodel == "spam11" {
                    candidates.insert(0, format!("s35_{submodel}_q{q_name}"));
                }
                for name in candidates {
                    if let Some(v) = out.remove(&name) {
                        ordered.push((name, v));
                    }
                }
            }
        }
    }
    // not expected, kept to not lose any submodel
    let mut rest: Vec<_> = out.into_iter().collect();
    rest.sort_by(|a, b| a.0.cmp(&b.0));
    ordered.extend(rest);
    ordered
}

/// Extract spatial rich model (SRM) features from a 2D array of u8 values.
///
/// Parameters
/// ----------
/// x1 : np.ndarray
///     A 2D NumPy array of unsigned 8-bit integers representing the image.
/// qs : list of list of float
///     Quantization steps of the five residual families.
/// q_names : list of list of str
///     Names of the quantization steps, used in the submodel names.
///
/// Returns
/// -------
/// dict
///     Symmetrized submodels, as float32 arrays, in the order of the Matlab implementation.
#[pyfunction]
#[pyo3(signature = (x1, qs, q_names))]
fn extract<'py>(py: Python<'py>, x1: PyReadonlyArray2<'py, u8>, qs: Vec<Vec<f64>>, q_names: Vec<Vec<String>>) -> PyResult<Bound<'py, PyDict>> {
    if qs.len() > 5 || qs.len() != q_names.len() || qs.iter().zip(&q_names).any(|(a, b)| a.len() != b.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("invalid quantization steps"));
    }
    // row-major image
    let view = x1.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    if h < 5 || w < 5 {
        return Err(pyo3::exceptions::PyValueError::new_err("image must be at least 5x5"));
    }
    let image = view.as_standard_layout();
    let x = image.as_slice().unwrap();
    // compute without holding the GIL, so that threads can extract features in parallel
    let submodels = py.detach(|| compute(x, h, w, &qs, &q_names));

    let features = PyDict::new(py);
    for (name, f) in submodels {
        features.set_item(name, PyArray1::from_vec(py, f))?;
    }
    Ok(features)
}

#[pymodule]
pub fn srm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
