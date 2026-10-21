
use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use numpy::{PyArray1, PyReadonlyArray2};

mod crm;
mod dctr;
mod gfr;
mod spam;
mod srm;

#[pymodule]
fn _sealwatch(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	let spam_mod = PyModule::new(py, "spam")?;
    spam::spam(py, &spam_mod)?;
    m.add_submodule(&spam_mod)?;

    let srm_mod = PyModule::new(py, "srm")?;
    srm::srm(py, &srm_mod)?;
    m.add_submodule(&srm_mod)?;

    let crm_mod = PyModule::new(py, "crm")?;
    crm::crm(py, &crm_mod)?;
    m.add_submodule(&crm_mod)?;

    let dctr_mod = PyModule::new(py, "dctr")?;
    dctr::dctr(py, &dctr_mod)?;
    m.add_submodule(&dctr_mod)?;

    let gfr_mod = PyModule::new(py, "gfr")?;
    gfr::gfr(py, &gfr_mod)?;
    m.add_submodule(&gfr_mod)?;

    // m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
