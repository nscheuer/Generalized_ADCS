//! PyO3 adapter for the Python-independent bias implementation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bias::Bias as CoreBias;

pub(super) fn numpy_array(py: Python<'_>, values: &[f64], shape: &[usize]) -> PyResult<PyObject> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (values.to_vec(),))?;
    let shape = PyTuple::new(py, shape);
    Ok(array.call_method1("reshape", (shape,))?.into())
}

pub(super) fn array_values(py: Python<'_>, value: &PyAny) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("atleast_1d", (value,))?;
    let array = numpy.call_method1("asarray", (array, "float64"))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let values = array
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?;
    Ok((values, shape))
}

fn broadcast_arrays(py: Python<'_>, values: &[&PyAny]) -> PyResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let numpy = py.import("numpy")?;
    let normalized: Vec<PyObject> = values
        .iter()
        .map(|value| numpy.call_method1("atleast_1d", (*value,)).map(Into::into))
        .collect::<PyResult<_>>()?;
    let tuple = PyTuple::new(py, normalized.iter().map(|value| value.as_ref(py)));
    let arrays = numpy.getattr("broadcast_arrays")?.call1(tuple)?;
    let mut output = Vec::with_capacity(values.len());
    let mut output_shape = Vec::new();
    for index in 0..values.len() {
        let array = arrays.get_item(index)?;
        if index == 0 {
            output_shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
        }
        output.push(
            array
                .call_method0("ravel")?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?,
        );
    }
    Ok((output, output_shape))
}

fn core_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

#[pyclass(name = "Bias")]
pub struct PyBias {
    inner: CoreBias,
}

#[pymethods]
impl PyBias {
    #[new]
    #[pyo3(signature = (bias=None, std_bias=None, bounds=None))]
    fn new(
        py: Python<'_>,
        bias: Option<PyObject>,
        std_bias: Option<PyObject>,
        bounds: Option<PyObject>,
    ) -> PyResult<Self> {
        let zero = 0.0_f64.into_py(py);
        let bias = bias.unwrap_or_else(|| zero.clone_ref(py));
        let std_bias = std_bias.unwrap_or_else(|| zero.clone_ref(py));
        let (lower, upper) = match bounds {
            Some(bounds) => bounds.as_ref(py).extract::<(PyObject, PyObject)>()?,
            None => (f64::NEG_INFINITY.into_py(py), f64::INFINITY.into_py(py)),
        };
        let (arrays, shape) = broadcast_arrays(
            py,
            &[
                bias.as_ref(py),
                std_bias.as_ref(py),
                lower.as_ref(py),
                upper.as_ref(py),
            ],
        )?;
        let mut arrays = arrays.into_iter();
        let inner = CoreBias::new(
            arrays.next().unwrap(),
            arrays.next().unwrap(),
            arrays.next().unwrap(),
            arrays.next().unwrap(),
            shape,
        )
        .map_err(core_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn bias_value(&self, py: Python<'_>) -> PyResult<PyObject> {
        numpy_array(py, &self.inner.bias, &self.inner.shape)
    }

    #[setter]
    fn set_bias(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let (values, shape) = array_values(py, value)?;
        if values.len() != self.inner.bias.len() {
            return Err(PyValueError::new_err(
                "bias must keep the same number of values",
            ));
        }
        self.inner.bias = values;
        self.inner.shape = shape;
        Ok(())
    }

    #[getter]
    fn std_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        numpy_array(py, &self.inner.std_bias, &self.inner.shape)
    }

    #[setter]
    fn set_std_bias(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let (values, shape) = array_values(py, value)?;
        if values.len() != self.inner.std_bias.len() {
            return Err(PyValueError::new_err(
                "std_bias must keep the same number of values",
            ));
        }
        self.inner.std_bias = values;
        self.inner.shape = shape;
        Ok(())
    }

    #[getter]
    fn estimated_std_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        numpy_array(py, &self.inner.estimated_std_bias, &self.inner.shape)
    }

    #[setter]
    fn set_estimated_std_bias(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let (values, shape) = array_values(py, value)?;
        if values.len() != self.inner.estimated_std_bias.len() {
            return Err(PyValueError::new_err(
                "estimated_std_bias must keep the same number of values",
            ));
        }
        self.inner.estimated_std_bias = values;
        self.inner.shape = shape;
        Ok(())
    }

    #[getter]
    fn last_bias_time(&self) -> f64 {
        self.inner.last_bias_time
    }

    #[setter]
    fn set_last_bias_time(&mut self, value: f64) {
        self.inner.last_bias_time = value;
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<PyObject> {
        let lower = numpy_array(py, &self.inner.lower_bound, &self.inner.shape)?;
        let upper = numpy_array(py, &self.inner.upper_bound, &self.inner.shape)?;
        Ok(PyTuple::new(py, [lower, upper]).into())
    }

    #[setter]
    fn set_bounds(&mut self, py: Python<'_>, value: &PyAny) -> PyResult<()> {
        let (lower, upper) = value.extract::<(PyObject, PyObject)>()?;
        let (arrays, _) = broadcast_arrays(py, &[lower.as_ref(py), upper.as_ref(py)])?;
        let mut arrays = arrays.into_iter();
        let lower = arrays.next().unwrap();
        let upper = arrays.next().unwrap();
        if lower.len() != self.inner.bias.len() || upper.len() != self.inner.bias.len() {
            return Err(PyValueError::new_err("bounds must match the bias shape"));
        }
        if lower.iter().zip(&upper).any(|(lo, hi)| lo > hi) {
            return Err(PyValueError::new_err(
                "Each lower bound must be <= the corresponding upper bound.",
            ));
        }
        self.inner.lower_bound = lower;
        self.inner.upper_bound = upper;
        Ok(())
    }

    fn __bool__(&self) -> bool {
        self.inner.is_active()
    }

    fn copy(&self) -> Self {
        Self {
            inner: self.inner.copy(),
        }
    }

    #[pyo3(signature = (length, *, owner="sensor"))]
    fn broadcast_to(&self, length: usize, owner: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.broadcast_to(length, owner).map_err(core_error)?,
        })
    }

    #[pyo3(signature = (dt=1.0, *, form="full"))]
    fn process_covariance(&self, py: Python<'_>, dt: f64, form: &str) -> PyResult<PyObject> {
        self.covariance(
            py,
            self.inner.process_covariance(dt).map_err(core_error)?,
            form,
            "bias",
        )
    }

    #[pyo3(signature = (*, form="full"))]
    fn process_psd(&self, py: Python<'_>, form: &str) -> PyResult<PyObject> {
        self.covariance(py, self.inner.process_psd(), form, "bias_rate")
    }

    #[pyo3(signature = (j2000, rng=None))]
    fn _update_bias(&mut self, py: Python<'_>, j2000: f64, rng: Option<PyObject>) -> PyResult<()> {
        if rng.is_none() {
            return self.inner.update_bias_default(j2000).map_err(core_error);
        }
        if self.inner.last_bias_time.is_nan() || j2000 <= self.inner.last_bias_time {
            return self
                .inner
                .update_bias_with_normals(j2000, &[])
                .map_err(core_error);
        }
        let rng = rng.unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("size", self.inner.bias.len())?;
        let draws = rng
            .as_ref(py)
            .call_method("normal", (0.0, 1.0), Some(kwargs))?;
        let (normals, _) = array_values(py, draws)?;
        self.inner
            .update_bias_with_normals(j2000, &normals)
            .map_err(core_error)
    }

    #[pyo3(name = "get_bias")]
    fn current_bias(&self, py: Python<'_>, j2000: f64) -> PyResult<PyObject> {
        let _ = j2000;
        if self.inner.bias.len() == 1 {
            Ok(self.inner.bias[0].into_py(py))
        } else {
            numpy_array(py, self.inner.get_bias(j2000), &self.inner.shape)
        }
    }

    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(
        &self,
        py: Python<'_>,
        value: &PyAny,
        dt: f64,
        rng: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let _ = (dt, rng);
        let (values, shape) = array_values(py, value)?;
        let result = self.inner.apply(&values).map_err(core_error)?;
        numpy_array(py, &result, &shape)
    }
}

impl PyBias {
    fn covariance(
        &self,
        py: Python<'_>,
        matrix: Vec<Vec<f64>>,
        form: &str,
        coordinates: &str,
    ) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("form", form)?;
        kwargs.set_item("coordinates", coordinates)?;
        py.import("ADCS.covariance")?
            .getattr("Covariance")?
            .call((matrix,), Some(kwargs))
            .map(Into::into)
    }
}
