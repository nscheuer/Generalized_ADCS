#![allow(non_local_definitions)]
use pyo3::prelude::*;

pub mod attitude_solver;
mod attitude_solver_py;
pub mod coupled_solver;
mod coupled_solver_py;
pub mod orbital_solver;
mod orbital_solver_py;
pub mod solver;
mod solver_py;

pub use coupled_solver::CoupledSolverConfig;
pub use solver::{SolverConfig, SolverMode};

#[pymodule]
fn _solver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<solver_py::PySolver>()?;
    m.add_class::<coupled_solver_py::PyCoupledSolver>()?;
    m.add_class::<attitude_solver_py::PyAttitudeSolver>()?;
    m.add_class::<orbital_solver_py::PyOrbitalSolver>()?;
    Ok(())
}
