use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList};
use pyo3::PyTypeInfo;

use super::adcs_subsystem::ADCSSubsystem as NativeADCSSubsystem;
use super::estimator::attitude_estimator::PyAttitudeEstimator;

fn positive_integer(value: &PyAny) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err("N must be a positive integer"));
    }
    let value = value
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("N must be a positive integer"))?;
    if value < 1 {
        return Err(PyValueError::new_err("N must be a positive integer"));
    }
    usize::try_from(value).map_err(|_| PyValueError::new_err("N must be a positive integer"))
}

fn collection(py: Python<'_>, value: Option<PyObject>) -> PyResult<PyObject> {
    match value {
        Some(values) => {
            let items: Vec<PyObject> = values.as_ref(py).extract()?;
            Ok(PyList::new(py, items).into())
        },
        None => Ok(PyList::empty(py).into()),
    }
}

fn items(py: Python<'_>, value: &PyObject) -> PyResult<Vec<PyObject>> {
    value.as_ref(py).extract()
}

#[pyclass(name = "ADCSSubsystem")]
pub struct ADCSSubsystem {
    inner: NativeADCSSubsystem<PyObject, PyObject, PyObject, PyObject>,
    reaction_wheels: PyObject,
    magnetorquers: PyObject,
    sensors: PyObject,
    estimator_object: Option<PyObject>,
}

#[pymethods]
impl ADCSSubsystem {
    #[new]
    #[pyo3(signature = (reaction_wheels=None, magnetorquers=None, sensors=None, N=1, estimator=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        reaction_wheels: Option<PyObject>,
        magnetorquers: Option<PyObject>,
        sensors: Option<PyObject>,
        #[pyo3(from_py_with = "positive_integer")] N: usize,
        estimator: Option<PyObject>,
    ) -> PyResult<Self> {
        let n = N;
        if let Some(ref object) = estimator {
            if !object
                .as_ref(py)
                .is_instance(PyAttitudeEstimator::type_object(py))?
            {
                return Err(PyTypeError::new_err(
                    "estimator must be an AttitudeEstimator",
                ));
            }
        }
        let reaction_wheels = collection(py, reaction_wheels)?;
        let magnetorquers = collection(py, magnetorquers)?;
        let sensors = collection(py, sensors)?;
        let wheel_items = items(py, &reaction_wheels)?;
        let torquer_items = items(py, &magnetorquers)?;
        let sensor_items = items(py, &sensors)?;
        let inner = NativeADCSSubsystem::with_components(
            wheel_items,
            torquer_items,
            sensor_items,
            n,
            estimator.as_ref().map(|value| value.clone_ref(py)),
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            reaction_wheels,
            magnetorquers,
            sensors,
            estimator_object: estimator,
        })
    }

    #[getter]
    fn reaction_wheels(&self, py: Python<'_>) -> PyObject {
        PyList::new(py, items(py, &self.reaction_wheels).unwrap_or_default()).into()
    }
    #[setter]
    fn set_reaction_wheels(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        self.inner.reaction_wheels = items(py, &value)?;
        self.reaction_wheels = value;
        Ok(())
    }

    #[getter]
    fn magnetorquers(&self, py: Python<'_>) -> PyObject {
        PyList::new(py, items(py, &self.magnetorquers).unwrap_or_default()).into()
    }
    #[setter]
    fn set_magnetorquers(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        self.inner.magnetorquers = items(py, &value)?;
        self.magnetorquers = value;
        Ok(())
    }

    #[getter]
    fn sensors(&self, py: Python<'_>) -> PyObject {
        PyList::new(py, items(py, &self.sensors).unwrap_or_default()).into()
    }
    #[setter]
    fn set_sensors(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        self.inner.sensors = items(py, &value)?;
        self.sensors = value;
        Ok(())
    }

    #[getter]
    #[allow(non_snake_case)]
    fn N(&self) -> usize {
        self.inner.N
    }
    #[setter]
    #[allow(non_snake_case)]
    fn set_N(&mut self, value: &PyAny) -> PyResult<()> {
        self.inner.N = positive_integer(value)?;
        Ok(())
    }

    #[getter]
    fn estimator(&self, py: Python<'_>) -> PyObject {
        self.estimator_object
            .as_ref()
            .map(|value| value.clone_ref(py))
            .unwrap_or_else(|| py.None())
    }
    #[setter]
    fn set_estimator(&mut self, py: Python<'_>, value: &PyAny) {
        self.estimator_object = if value.is_none() {
            None
        } else {
            Some(value.into_py(py))
        };
        self.inner.estimator = self
            .estimator_object
            .as_ref()
            .map(|object| object.clone_ref(py));
    }
}
