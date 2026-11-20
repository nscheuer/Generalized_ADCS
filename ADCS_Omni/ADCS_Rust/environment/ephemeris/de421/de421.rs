use pyo3::types::PyTuple;

#[pyclass(subclass)]
struct EphemerisModel {
    kernel: Kernel,
}

#[pymethods]
impl EphemerisModel {
    #[new]
    fn new(py: Python<'_>, segments: &PyAny, times: &PyAny, data: &PyAny) -> PyResult<Self> {
        Ok(Self {
            kernel: Kernel::from_arrays(py, segments, times, data)?,
        })
    }

    #[getter(_segments)]
    fn segments(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rows: Vec<Vec<i64>> = self.kernel.segments.iter().map(|s| {
            vec![s.target as i64, s.center as i64, s.kind as i64, s.offset as i64, s.size as i64]
        }).collect();
        Ok(py.import("numpy")?.call_method1("asarray", (rows,))?.call_method1("astype", ("int64",))?.into())
    }

    #[getter(_times)]
    fn times(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rows: Vec<Vec<f64>> = self.kernel.segments.iter().map(|s| vec![s.start, s.end]).collect();
        Ok(py.import("numpy")?.call_method1("asarray", (rows,))?.into())
    }

    #[getter(_data)]
    fn data(&self, py: Python<'_>) -> PyObject {
        self.kernel.data.to_pyarray(py).into_py(py)
    }

    fn position(&self, py: Python<'_>, body_id: i32, epoch_jd: f64) -> PyResult<PyObject> {
        let index = match body_id {
            0 => 10,
            1 => 11,
            2 => 9,
            _ => return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")),
        };
        Ok(self.kernel.planetary_state(index, epoch_jd)?.position.to_pyarray(py).into_py(py))
    }

    fn velocity(&self, py: Python<'_>, body_id: i32, epoch_jd: f64) -> PyResult<PyObject> {
        let index = match body_id {
            0 => 10,
            1 => 11,
            2 => 9,
            _ => return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")),
        };
        Ok(self.kernel.planetary_state(index, epoch_jd)?.velocity.to_pyarray(py).into_py(py))
    }

    #[staticmethod]
    fn eclipse_state(
        py: Python<'_>,
        spacecraft_position_m: &PyAny,
        sun_position_m: &PyAny,
        occulting_body_radius_m: f64,
        sun_radius_m: f64,
    ) -> PyResult<(f64, bool, bool)> {
        eclipse_state_py(py, spacecraft_position_m, sun_position_m, occulting_body_radius_m, sun_radius_m)
    }
}

#[pyclass(extends = EphemerisModel)]
struct DE421;

#[pymethods]
impl DE421 {
    #[classattr]
    const DATA_FILENAME: &'static str = "de421.bsp";

    #[new]
    #[pyo3(signature = (filepath=None))]
    fn new(py: Python<'_>, filepath: Option<&PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let path = if let Some(value) = filepath {
            PathBuf::from(value.str()?.to_str()?)
        } else {
            default_kernel_path(py)?
        };
        let kernel = Kernel::load(&path)?;
        Ok(PyClassInitializer::from(EphemerisModel { kernel }).add_subclass(Self))
    }

    #[getter]
    fn available_bodies(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, BODY_NAMES).into()
    }

    #[pyo3(signature = (target, epoch_jd_tdb, *, observer="earth"))]
    fn position_eci(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        target: &PyAny,
        epoch_jd_tdb: &PyAny,
        observer: &str,
    ) -> PyResult<PyObject> {
        let target = body_index(target)?;
        let observer = body_name_index(observer)?;
        let (epochs, shape, scalar) = epoch_values(py, epoch_jd_tdb)?;
        let base = slf.into_super();
        let kernel = &base.kernel;
        let values = epochs.iter().map(|epoch| {
            let t = kernel.planetary_state(target, *epoch)?.position;
            let o = kernel.planetary_state(observer, *epoch)?.position;
            Ok(sub(t, o))
        }).collect::<PyResult<Vec<_>>>()?;
        vector_result(py, &values, &shape, scalar)
    }

    #[pyo3(signature = (target, epoch_jd_tdb, *, observer="earth"))]
    fn velocity_eci(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        target: &PyAny,
        epoch_jd_tdb: &PyAny,
        observer: &str,
    ) -> PyResult<PyObject> {
        let target = body_index(target)?;
        let observer = body_name_index(observer)?;
        let (epochs, shape, scalar) = epoch_values(py, epoch_jd_tdb)?;
        let base = slf.into_super();
        let kernel = &base.kernel;
        let values = epochs.iter().map(|epoch| {
            let t = kernel.planetary_state(target, *epoch)?.velocity;
            let o = kernel.planetary_state(observer, *epoch)?.velocity;
            Ok(sub(t, o))
        }).collect::<PyResult<Vec<_>>>()?;
        vector_result(py, &values, &shape, scalar)
    }

    #[pyo3(signature = (target, epoch_jd_tdb, *, observer="earth"))]
    fn state_eci(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        target: &PyAny,
        epoch_jd_tdb: f64,
        observer: &str,
    ) -> PyResult<(PyObject, PyObject)> {
        let target = body_index(target)?;
        let observer = body_name_index(observer)?;
        let base = slf.into_super();
        let kernel = &base.kernel;
        let t = kernel.planetary_state(target, epoch_jd_tdb)?;
        let o = kernel.planetary_state(observer, epoch_jd_tdb)?;
        Ok((sub(t.position, o.position).to_pyarray(py).into_py(py), sub(t.velocity, o.velocity).to_pyarray(py).into_py(py)))
    }

    fn sun_position_eci(slf: PyRef<'_, Self>, py: Python<'_>, epoch_jd_tdb: &PyAny) -> PyResult<PyObject> {
        let target = pyo3::types::PyString::new(py, "sun");
        Self::position_eci(slf, py, target, epoch_jd_tdb, "earth")
    }
}

fn epoch_values(py: Python<'_>, value: &PyAny) -> PyResult<(Vec<f64>, Vec<usize>, bool)> {
    let array = py.import("numpy")?.call_method1("asarray", (value,))?.call_method1("astype", ("float64",))?;
    let scalar = array.getattr("ndim")?.extract::<usize>()? == 0;
    let shape = if scalar { Vec::new() } else { array.getattr("shape")?.extract::<Vec<usize>>()? };
    let values = if scalar {
        vec![array.call_method0("item")?.extract::<f64>()?]
    } else {
        array.call_method0("ravel")?.call_method0("tolist")?.extract::<Vec<f64>>()?
    };
    if values.iter().any(|v| !v.is_finite()) { return Err(PyValueError::new_err("epoch values must be finite")); }
    Ok((values, shape, scalar))
}

fn vector_result(py: Python<'_>, values: &[[f64; 3]], shape: &[usize], scalar: bool) -> PyResult<PyObject> {
    if scalar { return Ok(values[0].to_pyarray(py).into_py(py)); }
    let np = py.import("numpy")?;
    let rows: Vec<Vec<f64>> = values.iter().map(|v| v.to_vec()).collect();
    let array = np.call_method1("asarray", (rows,))?;
    let mut output_shape = shape.to_vec();
    output_shape.push(3);
    let tuple = PyTuple::new(py, output_shape);
    Ok(array.call_method1("reshape", (tuple,))?.into())
}
