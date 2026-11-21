use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
include!("solver.rs");
include!("coupled_solver.rs");
#[pymodule]
fn _solver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<CoupledSolver>()?;
    Ok(())
}
