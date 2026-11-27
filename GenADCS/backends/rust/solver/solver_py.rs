use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    #[pyo3(get, set)]
    tf: f64,
    #[pyo3(get, set)]
    progress: bool,
    #[pyo3(get, set)]
    progress_chunk_steps: usize,
    config: SolverConfig,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature=(dynamics=None,orbital=None,attitude=None,coupled=None,tf=None,progress=true,progress_chunk_steps=500))]
    fn new(
        py: Python<'_>,
        dynamics: Option<PyObject>,
        orbital: Option<PyObject>,
        attitude: Option<PyObject>,
        coupled: Option<PyObject>,
        tf: Option<f64>,
        progress: bool,
        progress_chunk_steps: usize,
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
        if progress_chunk_steps == 0 {
            return Err(PyValueError::new_err(
                "progress_chunk_steps must be a positive integer",
            ));
        }
        let final_time = tf.unwrap_or(0.0);
        if config.mode == super::solver::SolverMode::Split {
            if tf.is_none() || !final_time.is_finite() || final_time < 0.0 {
                return Err(PyValueError::new_err(
                    "separate solvers require a finite nonnegative tf",
                ));
            }
            let orbital_dt: f64 = orbital.as_ref(py).getattr("dt")?.extract()?;
            let attitude_dt: f64 = attitude.as_ref(py).getattr("dt")?.extract()?;
            if orbital_dt != attitude_dt {
                return Err(PyValueError::new_err(
                    "separate propagators require matching dt",
                ));
            }
        }
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
            tf: final_time,
            progress,
            progress_chunk_steps,
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
