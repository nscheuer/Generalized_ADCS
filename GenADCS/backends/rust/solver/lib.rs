#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod coupled_solver;
pub mod solver;
mod solver_py;

pub use coupled_solver::CoupledSolverConfig;
pub use solver::{SolverConfig, SolverMode};

#[pymodule]
fn _solver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<solver_py::PySolver>()?;
    m.add_class::<solver_py::PyCoupledSolver>()?;
    Ok(())
}
