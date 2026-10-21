use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray5};

use crate::dctr::{merged_histograms, merged_phases};

/// Valid convolution with the 8x8 kernel k (row-major),
/// out[i, j] = sum_uv x[i + u, j + v] k[7 - u, 7 - v], of shape [h - 7, w - 7].
fn convolve(x: &[f64], w: usize, h: usize, k: &[f64]) -> Vec<f64> {
    let rw = w - 7;
    let mut out = vec![0f64; (h - 7) * rw];
    for (i, row) in out.chunks_exact_mut(rw).enumerate() {
        // terms in the order of u and v, rows at a time
        for u in 0..8 {
            let x_row = &x[(i + u) * w..(i + u + 1) * w];
            for v in 0..8 {
                let kv = k[(7 - u) * 8 + 7 - v];
                for (o, &p) in row.iter_mut().zip(&x_row[v..v + rw]) {
                    *o += p * kv;
                }
            }
        }
    }
    out
}

/// GFR features of the image of shape [h, w], with kernels of shape [phases, scales, rotations, 8, 8],
/// in the order of the Python implementation.
fn compute(x: &[f64], h: usize, w: usize, kernels: &[f64], shape: [usize; 3], qs: &[f64], t: usize) -> Vec<(String, Vec<f64>)> {
    let groups = merged_phases();
    let [num_phases, num_scales, num_rotations] = shape;
    // histograms of each kernel, at [(phase * num_scales + scale) * num_rotations + rotation]
    let histograms: Vec<Vec<Vec<f64>>> = kernels.chunks_exact(64).enumerate()
        .map(|(idx, k)| {
            let scale = idx / num_rotations % num_scales;
            merged_histograms(&convolve(x, w, h, k), w - 7, qs[scale], t, &groups)
        })
        .collect();

    // merge the histograms of rotations theta and pi - theta
    let mut features = Vec::new();
    for phase in 0..num_phases {
        for scale in 0..num_scales {
            let at = |rotation: usize| &histograms[(phase * num_scales + scale) * num_rotations + rotation];
            for rotation in 0..=num_rotations / 2 {
                for g in 0..groups.len() {
                    let hist = if rotation == 0 || rotation == num_rotations / 2 {
                        at(rotation)[g].clone()
                    } else {
                        at(rotation)[g].iter().zip(&at(num_rotations - rotation)[g]).map(|(a, b)| (a + b) / 2.).collect()
                    };
                    features.push((format!("{phase}_{scale}_{rotation}_{g}"), hist));
                }
            }
        }
    }
    features
}

/// Extract GFR features from a decompressed image.
///
/// Parameters
/// ----------
/// x : np.ndarray
///     2D float64 array, decompressed image.
/// kernels : np.ndarray
///     Gabor kernels of shape [phases, scales, rotations, 8, 8], as sealwatch.gfr.compute_gabor_kernels().
/// quantization_steps : np.ndarray
///     Quantization step for each scale.
/// T : int
///     Truncation threshold.
///
/// Returns
/// -------
/// dict
///     Histograms by phase, scale, merged rotation and merged phases, in the order of the Python implementation.
#[pyfunction]
#[pyo3(signature = (x, kernels, quantization_steps, T))]
#[allow(non_snake_case)]
fn extract<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    kernels: PyReadonlyArray5<'py, f64>,
    quantization_steps: PyReadonlyArray1<'py, f64>,
    T: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let view = x.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    if h < 8 || w < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("image must be at least 8x8"));
    }
    let kernels_view = kernels.as_array();
    let ks = kernels_view.shape();
    if ks[3..] != [8, 8] || ks[2] % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernels must be 8x8, with an even number of rotations"));
    }
    let qs = quantization_steps.as_array().to_vec();
    if qs.len() != ks[1] {
        return Err(pyo3::exceptions::PyValueError::new_err("expected one quantization step per scale"));
    }
    // row-major
    let image = view.as_standard_layout();
    let x = image.as_slice().unwrap();
    let kernels_std = kernels_view.as_standard_layout();
    let k = kernels_std.as_slice().unwrap();
    let shape = [ks[0], ks[1], ks[2]];
    // compute without holding the GIL, so that threads can extract features in parallel
    let features = py.detach(|| compute(x, h, w, k, shape, &qs, T));

    let out = PyDict::new(py);
    for (name, f) in features {
        out.set_item(name, PyArray1::from_vec(py, f))?;
    }
    Ok(out)
}

#[pymodule]
pub fn gfr(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
