use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
include!("_coupled_dynamics.rs");
include!("_dop853_coefficients.rs");
include!("coupled_rk4_propagator.rs");
include!("coupled_dop853_propagator.rs");
include!("integrate.rs");
#[pymodule]
fn _coupled(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(derivative, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_step, m)?)?;
    m.add_function(wrap_pyfunction!(dop853_attempt, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_dop853, m)?)?;
    Ok(())
}
