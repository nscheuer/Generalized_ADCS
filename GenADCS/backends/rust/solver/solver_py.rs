use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::coupled_solver::CoupledSolverConfig;
use super::solver::SolverConfig;

#[pyclass(name = "Solver")]
pub struct PySolver {
    #[pyo3(get, set)]
    dynamics: PyObject,
    #[pyo3(get, set)]
    orbital: PyObject,
    #[pyo3(get, set)]
    attitude: PyObject,
    #[pyo3(get, set)]
    coupled: PyObject,
    config: SolverConfig,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature=(dynamics=None,orbital=None,attitude=None,coupled=None))]
    fn new(
        py: Python<'_>,
        dynamics: Option<PyObject>,
        orbital: Option<PyObject>,
        attitude: Option<PyObject>,
        coupled: Option<PyObject>,
    ) -> PyResult<Self> {
        let orbital = orbital.unwrap_or_else(|| py.None());
        let attitude = attitude.unwrap_or_else(|| py.None());
        let coupled = coupled.unwrap_or_else(|| py.None());
        let config = SolverConfig::new(
            !coupled.is_none(py),
            !orbital.is_none(py),
            !attitude.is_none(py),
        )
        .map_err(PyValueError::new_err)?;
        let dynamics = match dynamics {
            Some(value) => value,
            None => py
                .import("GenADCS.backends.rust.config")?
                .getattr("DynamicsConfig")?
                .call0()?
                .into_py(py),
        };
        Ok(Self {
            dynamics,
            orbital,
            attitude,
            coupled,
            config,
        })
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.config.mode {
            super::solver::SolverMode::Split => "split",
            super::solver::SolverMode::Coupled => "coupled",
        }
    }
}

#[pyclass(name = "CoupledSolver")]
pub struct PyCoupledSolver {
    propagator: PyObject,
    config: CoupledSolverConfig,
}

#[pymethods]
impl PyCoupledSolver {
    #[new]
    #[pyo3(signature=(propagator,tf,progress=false,progress_chunk_steps=500))]
    fn new(
        propagator: PyObject,
        tf: f64,
        progress: bool,
        progress_chunk_steps: usize,
    ) -> PyResult<Self> {
        let config = CoupledSolverConfig::new(tf, progress, progress_chunk_steps)
            .map_err(PyValueError::new_err)?;
        Ok(Self { propagator, config })
    }

    #[getter]
    fn propagator(&self, py: Python<'_>) -> PyObject {
        self.propagator.clone_ref(py)
    }
    #[setter]
    fn set_propagator(&mut self, value: PyObject) {
        self.propagator = value;
    }
    #[getter]
    fn tf(&self) -> f64 {
        self.config.final_time
    }
    #[setter]
    fn set_tf(&mut self, value: f64) -> PyResult<()> {
        self.config = CoupledSolverConfig::new(
            value,
            self.config.progress,
            self.config.progress_chunk_steps,
        )
        .map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[getter]
    fn progress(&self) -> bool {
        self.config.progress
    }
    #[setter]
    fn set_progress(&mut self, value: bool) {
        self.config.progress = value;
    }
    #[getter]
    fn progress_chunk_steps(&self) -> usize {
        self.config.progress_chunk_steps
    }
    #[setter]
    fn set_progress_chunk_steps(&mut self, value: usize) -> PyResult<()> {
        self.config = CoupledSolverConfig::new(self.config.final_time, self.config.progress, value)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[getter]
    fn dt(&self, py: Python<'_>) -> PyResult<f64> {
        self.propagator.as_ref(py).getattr("dt")?.extract()
    }
    fn step(
        &self,
        py: Python<'_>,
        state: PyObject,
        satellite: PyObject,
        environment: PyObject,
        dynamics: PyObject,
    ) -> PyResult<PyObject> {
        Ok(self
            .propagator
            .as_ref(py)
            .call_method1("step", (state, satellite, environment, dynamics))?
            .into_py(py))
    }
}

impl PyCoupledSolver {
    pub fn native_config(&self) -> CoupledSolverConfig {
        self.config
    }
}
