use crate::attitude_state::{quaternion, vector3};
use crate::internal_state::InternalState;
use crate::orbital_state::OrbitalState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct State {
    pub attitude: Py<crate::attitude_state::AttitudeState>,
    pub orbit: Py<OrbitalState>,
    pub internal: Py<InternalState>,
}

#[pymethods]
impl State {
    #[new]
    #[pyo3(signature = (attitude, orbit, internal=None))]
    fn new(
        py: Python<'_>,
        attitude: Py<crate::attitude_state::AttitudeState>,
        orbit: Py<OrbitalState>,
        internal: Option<Py<InternalState>>,
    ) -> PyResult<Self> {
        let internal = match internal {
            Some(value) => value,
            None => Py::new(
                py,
                InternalState {
                    wheel_momentum: Vec::new(),
                },
            )?,
        };
        Ok(Self {
            attitude,
            orbit,
            internal,
        })
    }
    #[getter]
    fn attitude(&self, py: Python<'_>) -> Py<crate::attitude_state::AttitudeState> {
        self.attitude.clone_ref(py)
    }
    #[setter]
    fn set_attitude(&mut self, py: Python<'_>, value: Py<crate::attitude_state::AttitudeState>) {
        self.attitude = value.clone_ref(py);
    }
    #[getter]
    fn orbit(&self, py: Python<'_>) -> Py<OrbitalState> {
        self.orbit.clone_ref(py)
    }
    #[setter]
    fn set_orbit(&mut self, py: Python<'_>, value: Py<OrbitalState>) {
        self.orbit = value.clone_ref(py);
    }
    #[getter]
    fn internal(&self, py: Python<'_>) -> Py<InternalState> {
        self.internal.clone_ref(py)
    }
    #[setter]
    fn set_internal(&mut self, py: Python<'_>, value: Py<InternalState>) {
        self.internal = value.clone_ref(py);
    }

    #[pyo3(signature = (wheel_count=None))]
    fn to_record(&self, py: Python<'_>, wheel_count: Option<isize>) -> PyResult<PyObject> {
        let attitude = self.attitude.borrow(py);
        let orbit = self.orbit.borrow(py);
        let internal = self.internal.borrow(py);
        if let Some(count) = wheel_count {
            if count < 0 {
                return Err(PyValueError::new_err("wheel_count must be nonnegative"));
            }
            if count as usize != internal.wheel_momentum.len() {
                return Err(PyValueError::new_err(format!(
                    "expected {count} reaction wheel momenta, got {}",
                    internal.wheel_momentum.len()
                )));
            }
        }
        let np = py.import("numpy")?;
        let fields = vec![
            ("quaternion", "float64", vec![4]),
            ("rates", "float64", vec![3]),
            ("position", "float64", vec![3]),
            ("velocity", "float64", vec![3]),
            ("momentum", "float64", vec![internal.wheel_momentum.len()]),
        ];
        let dtype = np.getattr("dtype")?.call1((fields,))?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", dtype)?;
        let record = np.call_method("empty", (1,), Some(kwargs))?;
        let qnorm = attitude
            .orientation
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if qnorm == 0.0 || !qnorm.is_finite() {
            return Err(PyValueError::new_err(
                "orientation quaternion must be finite and nonzero",
            ));
        }
        record.set_item("quaternion", vec![attitude.orientation.map(|v| v / qnorm)])?;
        record.set_item("rates", vec![attitude.rates])?;
        record.set_item(
            "position",
            vec![orbit.position.ok_or_else(|| {
                pyo3::exceptions::PyNotImplementedError::new_err(
                    "Keplerian to Cartesian conversion is not implemented",
                )
            })?],
        )?;
        record.set_item(
            "velocity",
            vec![orbit.velocity.ok_or_else(|| {
                pyo3::exceptions::PyNotImplementedError::new_err(
                    "Keplerian to Cartesian conversion is not implemented",
                )
            })?],
        )?;
        record.set_item("momentum", vec![internal.wheel_momentum.clone()])?;
        Ok(record.into())
    }

    #[staticmethod]
    #[pyo3(signature = (record, julian_date, ephemeris=None))]
    fn from_record(
        py: Python<'_>,
        record: &PyAny,
        julian_date: f64,
        ephemeris: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let value = if record.hasattr("dtype")? {
            record.get_item(0)?
        } else {
            record
        };
        let attitude = Py::new(
            py,
            crate::attitude_state::AttitudeState {
                orientation: quaternion(py, value.get_item("quaternion")?)?,
                rates: vector3(py, value.get_item("rates")?, "rates")?,
            },
        )?;
        let orbit = Py::new(
            py,
            OrbitalState::from_cartesian(
                py,
                value.get_item("position")?,
                value.get_item("velocity")?,
                julian_date,
                ephemeris,
            )?,
        )?;
        let momentum =
            crate::attitude_state::values(py, value.get_item("momentum")?, "wheel_momentum", None)?;
        let internal = Py::new(
            py,
            InternalState {
                wheel_momentum: momentum,
            },
        )?;
        Ok(Self {
            attitude,
            orbit,
            internal,
        })
    }
    #[staticmethod]
    fn from_python(py: Python<'_>, value: &PyAny) -> PyResult<Self> {
        let attitude = Py::new(
            py,
            crate::attitude_state::AttitudeState::from_python(py, value.getattr("attitude")?)?,
        )?;
        let orbit = Py::new(py, OrbitalState::from_python(py, value.getattr("orbit")?)?)?;
        let internal = Py::new(
            py,
            InternalState::from_python(py, value.getattr("internal")?)?,
        )?;
        Ok(Self {
            attitude,
            orbit,
            internal,
        })
    }
    fn to_python(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.import("GenADCS.backends.python.state")?.getattr("State")?;
        Ok(cls
            .call1((
                self.attitude.borrow(py).to_python(py)?,
                self.orbit.borrow(py).to_python(py)?,
                self.internal.borrow(py).to_python(py)?,
            ))?
            .into())
    }
}
