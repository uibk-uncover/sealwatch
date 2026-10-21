use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray2};


/// Groups of phases (row, col) in the 8x8 grid, whose histograms are merged by symmetry,
/// as get_symmetric_histogram_coordinates() in sealwatch/dctr/_dctr.py.
pub(crate) fn merged_phases() -> Vec<Vec<(usize, usize)>> {
    let mut groups = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let mut group = Vec::new();
            for p in [(i, j), (i, 8 - j), (8 - i, j), (8 - i, 8 - j)] {
                if p.0 < 8 && p.1 < 8 && !group.contains(&p) {
                    group.push(p);
                }
            }
            groups.push(group);
        }
    }
    groups
}

/// Valid convolution of the columns with kernel k,
/// out[i, j] = sum_u x[i + u, j] k[7 - u], of shape [h - 7, w].
fn convolve_vertical(x: &[f64], h: usize, w: usize, k: &[f64]) -> Vec<f64> {
    let mut out = vec![0f64; (h - 7) * w];
    for (i, row) in out.chunks_exact_mut(w).enumerate() {
        // terms in the order of u, rows at a time
        for u in 0..8 {
            for (o, &p) in row.iter_mut().zip(&x[(i + u) * w..(i + u + 1) * w]) {
                *o += p * k[7 - u];
            }
        }
    }
    out
}

/// Valid convolution of the rows with kernel k,
/// out[i, j] = sum_v x[i, j + v] k[7 - v], of shape [h, w - 7].
fn convolve_horizontal(x: &[f64], h: usize, w: usize, k: &[f64]) -> Vec<f64> {
    let rw = w - 7;
    let mut out = vec![0f64; h * rw];
    for (i, row) in out.chunks_exact_mut(rw).enumerate() {
        // terms in the order of v, rows at a time
        for v in 0..8 {
            for (o, &p) in row.iter_mut().zip(&x[i * w + v..i * w + v + rw]) {
                *o += p * k[7 - v];
            }
        }
    }
    out
}

/// Bin of a residual, its absolute value quantized by q, rounded as matlab_round, truncated to t.
///
/// Written without branches, as the values are unpredictable:
/// for a >= 0, matlab_round(a) is floor(a), plus one if the fraction is at least a half,
/// with the tolerance of np.isclose.
pub(crate) fn bin(r: f64, q: f64, t: f64) -> usize {
    let a = r.abs() / q;
    let floor = a.floor();
    let frac = a - floor;
    let half = (frac >= 0.5) | ((frac - 0.5).abs() <= 1e-8 + 1e-5 * 0.5);
    (floor + half as u8 as f64).min(t) as usize
}

/// Normalized histograms of the residuals r, of shape [rh, rw], quantized by q and truncated to t,
/// per group of phases (row, col) in the 8x8 grid.
pub(crate) fn merged_histograms(r: &[f64], rw: usize, q: f64, t: usize, groups: &[Vec<(usize, usize)>]) -> Vec<Vec<f64>> {
    // bins counted per phase (row, col) in the 8x8 grid, at counts[(phase * (t + 1) + bin]
    let mut counts = vec![0u64; 64 * (t + 1)];
    for (i, row) in r.chunks_exact(rw).enumerate() {
        let row_phase = (i % 8) * 8;
        for (j, &v) in row.iter().enumerate() {
            counts[(row_phase + j % 8) * (t + 1) + bin(v, q, t as f64)] += 1;
        }
    }
    // histograms of the phases merged by symmetry
    groups.iter().map(|phases| {
        let merged: Vec<u64> = (0..=t)
            .map(|b| phases.iter().map(|&(pr, pc)| counts[(pr * 8 + pc) * (t + 1) + b]).sum())
            .collect();
        let total: u64 = merged.iter().sum();
        merged.iter().map(|&c| c as f64 / total as f64).collect()
    }).collect()
}

/// DCTR features of the image of shape [h, w], with DCT matrix dct (row-major 8x8),
/// in the order of the Python implementation.
fn compute(x: &[f64], h: usize, w: usize, q: f64, t: usize, dct: &[f64]) -> Vec<(String, Vec<f64>)> {
    let groups = merged_phases();
    let (rh, rw) = (h - 7, w - 7);
    let mut features = Vec::with_capacity(64 * groups.len());
    for mode_row in 0..8 {
        // the DCT basis is the outer product of the rows of the DCT matrix,
        // the vertical convolution is shared by all modes in the row
        let vertical = convolve_vertical(x, h, w, &dct[mode_row * 8..(mode_row + 1) * 8]);
        for mode_col in 0..8 {
            let r = convolve_horizontal(&vertical, rh, w, &dct[mode_col * 8..(mode_col + 1) * 8]);
            for (g, hist) in merged_histograms(&r, rw, q, t, &groups).into_iter().enumerate() {
                features.push((format!("{}_{}", mode_row * 8 + mode_col, g), hist));
            }
        }
    }
    features
}

/// Extract DCTR features from a decompressed image.
///
/// Parameters
/// ----------
/// x : np.ndarray
///     2D float64 array, decompressed image with intensities in [-128, 127].
/// q : float
///     Quantization step.
/// T : int
///     Truncation threshold.
/// dct_mat : np.ndarray
///     8x8 DCT matrix, as sealwatch.tools.dct.compute_dct_mat().
///
/// Returns
/// -------
/// dict
///     Histograms by mode and merged phases, in the order of the Python implementation.
#[pyfunction]
#[pyo3(signature = (x, q, T, dct_mat))]
#[allow(non_snake_case)]
fn extract<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, q: f64, T: usize, dct_mat: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyDict>> {
    let view = x.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    if h < 8 || w < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("image must be at least 8x8"));
    }
    let dct_view = dct_mat.as_array();
    if dct_view.shape() != [8, 8] {
        return Err(pyo3::exceptions::PyValueError::new_err("DCT matrix must be 8x8"));
    }
    // row-major
    let image = view.as_standard_layout();
    let x = image.as_slice().unwrap();
    let dct = dct_view.as_standard_layout();
    let dct = dct.as_slice().unwrap();
    // compute without holding the GIL, so that threads can extract features in parallel
    let features = py.detach(|| compute(x, h, w, q, T, dct));

    let out = PyDict::new(py);
    for (name, f) in features {
        out.set_item(name, PyArray1::from_vec(py, f))?;
    }
    Ok(out)
}

#[pymodule]
pub fn dctr(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
