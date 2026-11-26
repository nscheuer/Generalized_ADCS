use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::orbital_solver::OrbitalSolverConfig;

#[pyclass(name = "OrbitalSolver")]
pub struct PyOrbitalSolver {
    propagator: PyObject,
    config: OrbitalSolverConfig,
}

#[pymethods]
impl PyOrbitalSolver {
    #[new]
    fn new(py: Python<'_>, propagator: PyObject) -> PyResult<Self> {
        let dt: f64 = propagator.as_ref(py).getattr("dt")?.extract()?;
        let config = OrbitalSolverConfig::new(dt).map_err(PyValueError::new_err)?;
        Ok(Self { propagator, config })
    }

    #[getter]
    fn propagator(&self, py: Python<'_>) -> PyObject {
        self.propagator.clone_ref(py)
    }

    #[setter]
    fn set_propagator(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        let dt: f64 = value.as_ref(py).getattr("dt")?.extract()?;
        self.config = OrbitalSolverConfig::new(dt).map_err(PyValueError::new_err)?;
        self.propagator = value;
        Ok(())
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.config.dt
    }
}
