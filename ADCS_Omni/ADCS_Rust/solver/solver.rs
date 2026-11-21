#[pyclass]
struct Solver {
    #[pyo3(get, set)]
    dynamics: PyObject,
    #[pyo3(get, set)]
    orbital: PyObject,
    #[pyo3(get, set)]
    attitude: PyObject,
    #[pyo3(get, set)]
    coupled: PyObject,
}
#[pymethods]
impl Solver {
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
        if (coupled.is_none(py) && (orbital.is_none(py) || attitude.is_none(py)))
            || (!coupled.is_none(py) && (!orbital.is_none(py) || !attitude.is_none(py)))
        {
            return Err(PyValueError::new_err(
                "provide a coupled solver or both orbital and attitude solvers",
            ));
        }
        let dynamics = match dynamics {
            Some(value) => value,
            None => py
                .import("ADCS_Omni.ADCS_Rust.config")?
                .getattr("DynamicsConfig")?
                .call0()?
                .into_py(py),
        };
        Ok(Self {
            dynamics,
            orbital,
            attitude,
            coupled,
        })
    }
}
