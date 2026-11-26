use crate::attitude_state_py::{array, vector3};
use crate::state_types::{KeplerianElements, Vector3};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct OrbitalState {
    pub position: Option<Vector3>,
    pub velocity: Option<Vector3>,
    pub julian_date: f64,
    pub ephemeris: Option<Py<PyAny>>,
    pub keplerian_elements: Option<KeplerianElements>,
    body_positions: HashMap<String, Py<PyAny>>,
}

#[pymethods]
impl OrbitalState {
    #[new]
    #[pyo3(signature = (_position, _velocity, julian_date, ephemeris=None))]
    fn new(
        py: Python<'_>,
        _position: Option<&PyAny>,
        _velocity: Option<&PyAny>,
        julian_date: f64,
        ephemeris: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Self::build(py, _position, _velocity, julian_date, ephemeris, None)
    }
    #[staticmethod]
    #[pyo3(signature = (position, velocity, julian_date, ephemeris=None))]
    pub fn from_cartesian(
        py: Python<'_>,
        position: &PyAny,
        velocity: &PyAny,
        julian_date: f64,
        ephemeris: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            Some(position),
            Some(velocity),
            julian_date,
            ephemeris,
            None,
        )
    }
    #[staticmethod]
    #[pyo3(signature = (semi_major_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, true_anomaly, julian_date, ephemeris=None))]
    pub fn from_keplerian(
        py: Python<'_>,
        semi_major_axis: f64,
        eccentricity: f64,
        inclination: f64,
        ascending_node: f64,
        argument_of_periapsis: f64,
        true_anomaly: f64,
        julian_date: f64,
        ephemeris: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            None,
            None,
            julian_date,
            ephemeris,
            Some([
                semi_major_axis,
                eccentricity,
                inclination,
                ascending_node,
                argument_of_periapsis,
                true_anomaly,
            ]),
        )
    }
    #[getter(_position)]
    fn get_position(&self, py: Python<'_>) -> PyResult<PyObject> {
        option_vec3(py, self.position)
    }
    #[setter(_position)]
    fn set_position(&mut self, py: Python<'_>, value: Option<&PyAny>) -> PyResult<()> {
        self.position = optional_vec3(py, value, "position")?;
        Ok(())
    }
    #[getter(_velocity)]
    fn get_velocity(&self, py: Python<'_>) -> PyResult<PyObject> {
        option_vec3(py, self.velocity)
    }
    #[setter(_velocity)]
    fn set_velocity(&mut self, py: Python<'_>, value: Option<&PyAny>) -> PyResult<()> {
        self.velocity = optional_vec3(py, value, "velocity")?;
        Ok(())
    }
    #[getter]
    pub fn position(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.cartesian_vectors()?.0)
    }
    #[getter]
    pub fn velocity(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.cartesian_vectors()?.1)
    }
    #[getter]
    fn julian_date(&self) -> f64 {
        self.julian_date
    }
    #[setter]
    fn set_julian_date(&mut self, value: f64) {
        self.julian_date = value;
    }
    #[getter]
    fn ephemeris(&self, py: Python<'_>) -> PyObject {
        self.ephemeris
            .as_ref()
            .map_or_else(|| py.None(), |v| v.clone_ref(py).into_py(py))
    }
    #[setter]
    fn set_ephemeris(&mut self, value: Option<Py<PyAny>>) {
        self.ephemeris = value;
    }
    #[getter(_keplerian_elements)]
    fn get_keplerian_elements(&self, py: Python<'_>) -> PyObject {
        self.keplerian_elements
            .map_or_else(|| py.None(), |values| values.into_py(py))
    }
    fn body_position(&mut self, py: Python<'_>, body: &PyAny) -> PyResult<PyObject> {
        let key = format!(
            "{}:{}",
            body.get_type().str()?.to_str()?,
            body.str()?.to_str()?
        );
        if let Some(value) = self.body_positions.get(&key) {
            return Ok(value.clone_ref(py).into_py(py));
        }
        let ephemeris = self
            .ephemeris
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("an ephemeris must be configured"))?;
        let body_id = if let Ok(name) = body.extract::<String>() {
            match name.to_lowercase().as_str() {
                "sun" => 0,
                "earth" => 1,
                "moon" => 2,
                other => return Err(PyKeyError::new_err(other.to_string())),
            }
        } else {
            body.extract::<i32>()?
        };
        let result = ephemeris
            .as_ref(py)
            .call_method1("position", (body_id, self.julian_date))?
            .into_py(py);
        self.body_positions.insert(key, result.clone_ref(py));
        Ok(result)
    }
    #[getter]
    fn sun_position(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.body_position(py, pyo3::types::PyString::new(py, "sun"))
    }
    #[getter]
    fn earth_position(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.body_position(py, pyo3::types::PyString::new(py, "earth"))
    }
    #[getter]
    fn moon_position(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.body_position(py, pyo3::types::PyString::new(py, "moon"))
    }
    #[staticmethod]
    pub fn from_python(py: Python<'_>, state: &PyAny) -> PyResult<Self> {
        let elements = state.getattr("_keplerian_elements")?;
        let jd = state.getattr("julian_date")?.extract::<f64>()?;
        let ephem = state.getattr("ephemeris")?;
        let ephem = if ephem.is_none() {
            None
        } else {
            Some(ephem.into_py(py))
        };
        if elements.is_none() {
            Self::build(
                py,
                Some(state.getattr("_position")?),
                Some(state.getattr("_velocity")?),
                jd,
                ephem,
                None,
            )
        } else {
            let el: Vec<f64> = elements.extract()?;
            let el: KeplerianElements = el
                .try_into()
                .map_err(|_| PyValueError::new_err("expected six Keplerian elements"))?;
            Self::build(py, None, None, jd, ephem, Some(el))
        }
    }
    pub fn to_python(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py
            .import("GenADCS.backends.python.state")?
            .getattr("OrbitalState")?;
        let ephem = self
            .ephemeris
            .as_ref()
            .map_or_else(|| py.None(), |v| v.clone_ref(py).into_py(py));
        if let Some(e) = self.keplerian_elements {
            Ok(cls
                .call_method1(
                    "from_keplerian",
                    (e[0], e[1], e[2], e[3], e[4], e[5], self.julian_date, ephem),
                )?
                .into())
        } else {
            Ok(cls
                .call1((
                    option_vec3(py, self.position)?,
                    option_vec3(py, self.velocity)?,
                    self.julian_date,
                    ephem,
                ))?
                .into())
        }
    }
}

impl OrbitalState {
    fn cartesian_vectors(&self) -> PyResult<(Vector3, Vector3)> {
        let core = crate::orbital_state::OrbitalState {
            position: self.position,
            velocity: self.velocity,
            julian_date: self.julian_date,
            keplerian_elements: self.keplerian_elements,
        };
        core.cartesian().map_err(PyValueError::new_err)
    }

    fn build(
        py: Python<'_>,
        p: Option<&PyAny>,
        v: Option<&PyAny>,
        jd: f64,
        ephemeris: Option<Py<PyAny>>,
        elements: Option<KeplerianElements>,
    ) -> PyResult<Self> {
        let position = optional_vec3(py, p, "position")?;
        let velocity = optional_vec3(py, v, "velocity")?;
        let core = match elements {
            Some(elements) => crate::orbital_state::OrbitalState::new_keplerian(elements, jd),
            None => crate::orbital_state::OrbitalState::new_cartesian(position, velocity, jd),
        }
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            position: core.position,
            velocity: core.velocity,
            julian_date: core.julian_date,
            ephemeris,
            keplerian_elements: core.keplerian_elements,
            body_positions: HashMap::new(),
        })
    }
}
fn optional_vec3(py: Python<'_>, value: Option<&PyAny>, name: &str) -> PyResult<Option<Vector3>> {
    value.map(|v| vector3(py, v, name)).transpose()
}
fn option_vec3(py: Python<'_>, value: Option<Vector3>) -> PyResult<PyObject> {
    match value {
        Some(v) => array(py, v),
        None => Ok(py.None()),
    }
}

#[cfg(test)]
#[path = "../../../tests/backends/rust/state/test_orbital_state.rs"]
mod tests;
