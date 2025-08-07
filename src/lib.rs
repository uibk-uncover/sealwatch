
use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use numpy::{PyArray1, PyReadonlyArray2};

mod spam;

#[pymodule]
fn _sealwatch(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	let spam_mod = PyModule::new(py, "spam")?;
    spam::spam(py, &spam_mod)?;
    m.add_submodule(&spam_mod)?;

    // m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
