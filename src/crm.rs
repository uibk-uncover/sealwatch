use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray3};

use crate::srm::{
    all1st, all2nd, all3rd, all_edge, quantize, residual,
    Dir, Plane, Submodels, FAMILIES, FAMILY_KERNELS,
};

/// Rounding as in Matlab, as sealwatch.tools.matlab.round, on float32, as NumPy computes it.
fn matlab_round_f32(v: f32) -> f32 {
    let a = v.abs();
    let floor = a.floor();
    let frac = a - floor;
    // np.isclose(frac, 0.5), in float32
    let half = frac >= 0.5 || (frac - 0.5).abs() <= (1e-8 + 1e-5 * 0.5) as f32;
    let rounded = if half { a.ceil() } else { floor };
    if v < 0.0 { -rounded } else { rounded }
}

/// Normalized 3rd-order co-occurrence of the three channels of each pixel, over all planes.
/// The direction is ignored, as in cooccurrence3_col in sealwatch/srm/cooccurrence.py.
fn cooc3(planes: &[&Plane], _dir: Dir, b: usize) -> Vec<f64> {
    // counted in four copies, pixel k into copy k mod 4,
    // so that consecutive pixels falling into the same bin do not wait for each other
    let mut copies = [vec![0u32; b * b * b], vec![0u32; b * b * b], vec![0u32; b * b * b], vec![0u32; b * b * b]];
    let index = |px: &[u8]| (px[0] as usize * b + px[1] as usize) * b + px[2] as usize;
    for p in planes {
        let mut pixels = p.v.chunks_exact(4 * 3);
        for four in &mut pixels {
            copies[0][index(&four[0..3])] += 1;
            copies[1][index(&four[3..6])] += 1;
            copies[2][index(&four[6..9])] += 1;
            copies[3][index(&four[9..12])] += 1;
        }
        for px in pixels.remainder().chunks_exact(3) {
            copies[0][index(px)] += 1;
        }
    }
    let counts: Vec<u64> = (0..b * b * b).map(|i| copies.iter().map(|c| c[i] as u64).sum()).collect();
    let total: u64 = counts.iter().sum();
    counts.iter().map(|&c| c as f64 / total as f64).collect()
}

/// Index of bin (i, j, k), each in [0, b), of a 3D co-occurrence.
fn idx3(i: usize, j: usize, k: usize, b: usize) -> usize {
    (i * b + j) * b + k
}

/// Sign and directional symmetrization of a spam co-occurrence, as symm3.
fn symm3(a: &[f32], t: usize) -> Vec<f32> {
    let b = 2 * t + 1;
    let m = b - 1;
    let mut done = vec![false; a.len()];
    let mut out = vec![0f32; 1 + 3 * t + 4 * t * t + 2 * t * t * t];
    out[0] = a[idx3(t, t, t, b)];
    let mut next = 1;
    for i in 0..b {
        for j in 0..b {
            for k in 0..b {
                if done[idx3(i, j, k, b)] || (i, j, k) == (t, t, t) {
                    continue;
                }
                out[next] = a[idx3(i, j, k, b)] + a[idx3(m - i, m - j, m - k, b)];
                done[idx3(i, j, k, b)] = true;
                done[idx3(m - i, m - j, m - k, b)] = true;
                if i != k && !done[idx3(k, j, i, b)] {
                    out[next] = out[next] + a[idx3(k, j, i, b)] + a[idx3(m - k, m - j, m - i, b)];
                    done[idx3(k, j, i, b)] = true;
                    done[idx3(m - k, m - j, m - i, b)] = true;
                }
                next += 1;
            }
        }
    }
    out
}

/// Sign and directional symmetrization of a min and max co-occurrence, as symm3_dir.
fn symm3_dir(fmin: &[f32], fmax: &[f32], t: usize) -> Vec<f32> {
    let b = 2 * t + 1;
    let m = b - 1;
    // sign symmetrization, min with flipped max
    let mut a = vec![0f32; fmin.len()];
    for i in 0..b {
        for j in 0..b {
            for k in 0..b {
                a[idx3(i, j, k, b)] = fmin[idx3(i, j, k, b)] + fmax[idx3(m - i, m - j, m - k, b)];
            }
        }
    }
    // directional symmetrization
    let mut done = vec![false; a.len()];
    let mut out = Vec::with_capacity(b * b * b - t * b * b);
    for i in 0..b {
        for j in 0..b {
            for k in 0..b {
                if i != k {
                    if !done[idx3(i, j, k, b)] {
                        out.push(a[idx3(i, j, k, b)] + a[idx3(k, j, i, b)]);
                        done[idx3(i, j, k, b)] = true;
                        done[idx3(k, j, i, b)] = true;
                    }
                } else {
                    out.push(a[idx3(i, j, k, b)]);
                    done[idx3(i, j, k, b)] = true;
                }
            }
        }
    }
    out
}

/// Names of the color submodels, as name_mapper in sealwatch/srm/crm.py.
const NAMES: &[(&str, &str)] = &[
    ("spam11", "spam11c"), ("spam12h", "spam12c"), ("spam14h", "spam14c"),
    ("min21", "min21c"), ("max21", "max21c"),
    ("min22h", "min22c"), ("max22h", "max22c"),
    ("min24", "min24c"), ("max24", "max24c"), ("min24h", "min24c"), ("max24h", "max24c"),
    ("min32", "min32c"), ("max32", "max32c"),
    ("min34", "min34c"), ("max34", "max34c"), ("min34h", "min34hvc"), ("max34h", "max34hvc"),
    ("min41", "min41c"), ("max41", "max41c"),
    ("min48h", "min48c"), ("max48h", "max48c"),
    ("min54", "min54c"), ("max54", "max54c"),
];

fn color_name(name: &str) -> &str {
    NAMES.iter().find(|(n, _)| *n == name).map_or(name, |(_, c)| c)
}

/// Symmetrized submodels of a family, in the order of their first occurrence in g.
fn post_process(g: Submodels, family: &str, q: &str, t: usize, out: &mut Vec<(String, Vec<f32>)>) {
    let g: Vec<(&str, Vec<f32>)> = g.into_iter()
        .map(|(name, f)| (color_name(name), f.iter().map(|&v| v as f32).collect()))
        .collect();
    let get = |name: &str| &g.iter().find(|(n, _)| *n == name).unwrap().1;
    for (name, f) in &g {
        if let Some(suffix) = name.strip_prefix("min").or_else(|| name.strip_prefix("max")) {
            let out_name = format!("{family}_minmax{suffix}_{q}");
            if !out.iter().any(|(n, _)| *n == out_name) {
                out.push((out_name, symm3_dir(get(&format!("min{suffix}")), get(&format!("max{suffix}")), t)));
            }
        } else if name.starts_with("spam") {
            out.push((format!("{family}_{name}_{q}"), symm3(f, t)));
        }
    }
}

/// Color rich model features, in the order of the Python implementation.
fn compute(x: &[u8], h: usize, w: usize, q: f64, q_name: &str, tc: i32) -> Vec<(String, Vec<f32>)> {
    let b = (2 * tc + 1) as usize;
    let cooc = |planes: &[&Plane], dir: Dir| cooc3(planes, dir, b);
    let multipliers = [1.0, 2.0, 3.0, 4.0, 12.0];
    let q_name = format!("q{q_name}");
    let mut out = Vec::new();
    for f in 0..5 {
        // division in float32, as the image is converted to float32
        let qc = (q * multipliers[f]) as f32;
        let y: Vec<Plane> = FAMILY_KERNELS[f].iter()
            .map(|k| quantize(&residual(x, h, w, 3, k), tc, |r| matlab_round_f32(r as f32 / qc) as i32))
            .collect();
        let g = match f {
            0 => all1st(&y, &cooc, false),
            1 => all2nd(&y, &cooc, false),
            2 => all3rd(&y, &cooc, false),
            _ => all_edge(&y, &cooc, false),
        };
        post_process(g, FAMILIES[f], &q_name, tc as usize, &mut out);
    }
    // spam11 of 3x3 (KB) and 5x5 (KV) united, at the end
    let take = |out: &mut Vec<(String, Vec<f32>)>, name: &str| {
        out.iter().position(|(n, _)| n == name).map(|i| out.remove(i).1)
    };
    let (k3, k5) = (format!("s3x3_spam11c_{q_name}"), format!("s5x5_spam11c_{q_name}"));
    if let (Some(f3), Some(f5)) = (take(&mut out, &k3), take(&mut out, &k5)) {
        out.push((format!("s35_spam11c_{q_name}"), [f3, f5].concat()));
    }
    out
}

/// Extract color rich model (CRM) features from a 3D array of u8 values.
///
/// Parameters
/// ----------
/// x1 : np.ndarray
///     A 3D NumPy array of unsigned 8-bit integers, of shape [height, width, 3].
/// q : float
///     Quantization step.
/// q_name : str
///     Name of the quantization step, used in the submodel names.
/// Tc : int
///     Truncation threshold of the color co-occurrences.
///
/// Returns
/// -------
/// dict
///     Symmetrized submodels, as float32 arrays, in the order of the Python implementation.
#[pyfunction]
#[pyo3(signature = (x1, q, q_name, Tc))]
#[allow(non_snake_case)]
fn extract<'py>(py: Python<'py>, x1: PyReadonlyArray3<'py, u8>, q: f64, q_name: String, Tc: i32) -> PyResult<Bound<'py, PyDict>> {
    let view = x1.as_array();
    let (h, w, c) = (view.shape()[0], view.shape()[1], view.shape()[2]);
    if c != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("image must have 3 channels"));
    }
    if h < 5 || w < 5 {
        return Err(pyo3::exceptions::PyValueError::new_err("image must be at least 5x5"));
    }
    if !(1..=7).contains(&Tc) {
        return Err(pyo3::exceptions::PyValueError::new_err("Tc must be in [1, 7]"));
    }
    // row-major image, channels next to each other
    let image = view.as_standard_layout();
    let x = image.as_slice().unwrap();
    // compute without holding the GIL, so that threads can extract features in parallel
    let submodels = py.detach(|| compute(x, h, w, q, &q_name, Tc));

    let features = PyDict::new(py);
    for (name, f) in submodels {
        features.set_item(name, PyArray1::from_vec(py, f))?;
    }
    Ok(features)
}

#[pymodule]
pub fn crm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
