
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray2};


/// Extract co-occurrence features from a 2D array of i16 values.
///
/// Parameters
/// ----------
/// array : np.ndarray
///     A 2D NumPy array of signed 16-bit integers representing the image.
/// t : int, optional
///     Truncation threshold. Defaults to 3.
///
/// Returns
/// -------
/// dict
///     Dictionary with normalized co-occurrence features:
///     - "straight": 1D NumPy array of straight co-occurrences
///     - "diagonal": 1D NumPy array of diagonal co-occurrences
#[pyfunction]
#[pyo3(signature = (array, t = 3))]
fn extract<'py>(py: Python<'py>, array: PyReadonlyArray2<'py, i16>, t: Option<i16>) -> PyResult<PyObject> {
    let input = array.as_array();
    let t = t.unwrap_or(3);

    // Collect coocurrences
    let D = (2 * t + 1) as usize;
    let mut cooc_straight = vec![vec![vec![0 as i32; D]; D]; D];
    let mut cooc_diagonal = vec![vec![vec![0 as i32; D]; D]; D];
    let h = input.shape()[0] as usize;
    let w = input.shape()[1] as usize;
    //
    for col in 1..(w-2) {
        for row in 0..h {
            //
            let l = (input[[row, col+1]] - input[[row, col+2]]).clamp(-t, t);
            let c = (input[[row, col]] - input[[row, col+1]]).clamp(-t, t);
            let r = (input[[row, col-1]] - input[[row, col]]).clamp(-t, t);
            //
            let x_idx = (l + t) as usize;
            let y_idx = (c + t) as usize;
            let z_idx = (r + t) as usize;
            //
            cooc_straight[z_idx][y_idx][x_idx] += 1;
            cooc_straight[D-x_idx-1][D-y_idx-1][D-z_idx-1] += 1;
        }
    }
    for col in 0..w {
        for row in 1..(h-2) {
            //
            let l = (input[[row+1, col]] - input[[row+2, col]]).clamp(-t, t);
            let c = (input[[row, col]] - input[[row+1, col]]).clamp(-t, t);
            let r = (input[[row-1, col]] - input[[row, col]]).clamp(-t, t);
            //
            let x_idx = (l + t) as usize;
            let y_idx = (c + t) as usize;
            let z_idx = (r + t) as usize;
            //
            cooc_straight[z_idx][y_idx][x_idx] += 1;
            cooc_straight[D-x_idx-1][D-y_idx-1][D-z_idx-1] += 1;
        }
    }
    for col in 1..(w-2) {
        for row in 1..(h-2) {
            //
            let l = (input[[row+1, col+1]] - input[[row+2, col+2]]).clamp(-t, t);
            let c = (input[[row, col]] - input[[row+1, col+1]]).clamp(-t, t);
            let r = (input[[row-1, col-1]] - input[[row, col]]).clamp(-t, t);
            //
            let x_idx = (l + t) as usize;
            let y_idx = (c + t) as usize;
            let z_idx = (r + t) as usize;
            //
            cooc_diagonal[z_idx][y_idx][x_idx] += 1;
            cooc_diagonal[D-x_idx-1][D-y_idx-1][D-z_idx-1] += 1;

            //
            let l = (input[[row, col+1]] - input[[row-1, col+2]]).clamp(-t, t);
            let c = (input[[row+1, col]] - input[[row, col+1]]).clamp(-t, t);
            let r = (input[[row+2, col-1]] - input[[row+1, col]]).clamp(-t, t);
            //
            let x_idx = (l + t) as usize;
            let y_idx = (c + t) as usize;
            let z_idx = (r + t) as usize;
            //
            cooc_diagonal[z_idx][y_idx][x_idx] += 1;
            cooc_diagonal[D-x_idx-1][D-y_idx-1][D-z_idx-1] += 1;
        }
    }

    // Flatten the 2D matrix to 1D
    let cooc_straight_flat: Vec<f64> = cooc_straight.iter()
        .flat_map(|m| m.iter())
        .flat_map(|v| v.iter())
        .map(|&val| val as f64)
        .collect();
    let cooc_diagonal_flat: Vec<f64> = cooc_diagonal.iter()
        .flat_map(|m| m.iter())
        .flat_map(|v| v.iter())
        .map(|&val| val as f64)
        .collect();

    // Create a 1D numpy array from the flattened data
    let total_straight: f64 = cooc_straight_flat.iter().sum();
    let cooc_straight_flat_norm: Vec<f64> = cooc_straight_flat.iter().map(|&v| v / total_straight).collect();
    let py_cooc_straight = PyArray1::from_vec(py, cooc_straight_flat_norm);
    let total_diagonal: f64 = cooc_diagonal_flat.iter().sum();
    let cooc_diagonal_flat_norm: Vec<f64> = cooc_diagonal_flat.iter().map(|&v| v / total_diagonal).collect();
    let py_cooc_diagonal = PyArray1::from_vec(py, cooc_diagonal_flat_norm);

    // Create a dictionary
    let features = PyDict::new(py);
    features.set_item("straight", py_cooc_straight)?;
    features.set_item("diagonal", py_cooc_diagonal)?;
    Ok(features.into())
}

#[pymodule]
pub fn spam_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}