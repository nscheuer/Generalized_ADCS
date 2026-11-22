fn dcmarg(py: Python<'_>, x: &PyAny) -> PyResult<M> {
    if x.hasattr("as_matrix")? {
        mat3(py, x.call_method0("as_matrix")?, "DCM")
    } else {
        mat3(py, x, "DCM")
    }
}
#[pyclass]
struct DCM {
    m: M,
}
#[pymethods]
impl DCM {
    #[new]
    #[pyo3(signature=(matrix=None))]
    fn new(py: Python<'_>, matrix: Option<&PyAny>) -> PyResult<Self> {
        let m = if let Some(v) = matrix {
            mat3(py, v, "DCM")?
        } else {
            M::identity()
        };
        if !proper_dcm(m) {
            return Err(PyValueError::new_err(
                "DCM must be a proper rotation matrix",
            ));
        }
        Ok(Self { m })
    }
    #[staticmethod]
    fn identity() -> Self {
        Self { m: M::identity() }
    }
    #[staticmethod]
    fn from_axis_angle(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<Self> {
        let m = axis_angle_to_dcm(py, axis, angle)?;
        Self::new(py, Some(m.as_ref(py)))
    }
    #[staticmethod]
    fn from_rotation_vector(py: Python<'_>, r: &PyAny) -> PyResult<Self> {
        let q = rotation_vector_to_quaternion(py, r)?;
        let m = _quaternion_to_dcm(py, q.as_ref(py))?;
        Self::new(py, Some(m.as_ref(py)))
    }
    #[staticmethod]
    fn from_quaternion(py: Python<'_>, q: &PyAny) -> PyResult<Self> {
        let q = quatarg(py, q)?;
        Self { m: qdcm(q) }.validate()
    }
    #[staticmethod]
    fn from_dcm(py: Python<'_>, m: &PyAny) -> PyResult<Self> {
        Self::new(py, Some(m))
    }
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pymat(py, &self.m))
    }
    #[getter]
    fn hamilton_components(&self, py: Python<'_>) -> PyResult<PyObject> {
        _dcm_to_quaternion(py, pymat(py, &self.m).as_ref(py))
    }
    #[getter]
    fn norm(&self) -> f64 {
        let t = tr(self.m);
        (t * self.m - M::identity()).norm()
    }
    #[getter]
    fn orthogonality_error(&self) -> f64 {
        let t = tr(self.m);
        (t * self.m - M::identity()).norm()
    }
    fn as_matrix(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pymat(py, &self.m))
    }
    fn to_dcm(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pymat(py, &self.m))
    }
    fn to_quaternion(&self, py: Python<'_>) -> PyResult<Quaternion> {
        let q = _dcm_to_quaternion(py, pymat(py, &self.m).as_ref(py))?;
        Quaternion::new(py, Some(q.as_ref(py)), "Hamilton")
    }
    fn normalized(&self, py: Python<'_>) -> PyResult<Self> {
        let m = normalize_dcm(py, pymat(py, &self.m).as_ref(py))?;
        Self::new(py, Some(m.as_ref(py)))
    }
    fn inverse(&self) -> Self {
        Self { m: tr(self.m) }
    }
    fn conjugate(&self) -> Self {
        self.inverse()
    }
    fn multiply(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        let b = dcmarg(py, other)?;
        Self { m: mm(self.m, b) }.validate()
    }
    fn __mul__(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        self.multiply(py, other)
    }
    fn rotate(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        pyvec(py, mv(self.m, fixed(py, v, "vector")?))
    }
    fn inertial_to_body(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        self.rotate(py, v)
    }
    fn body_to_inertial(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        pyvec(py, mv(tr(self.m), fixed(py, v, "vector")?))
    }
    fn relative_to(&self, py: Python<'_>, r: &PyAny) -> PyResult<Self> {
        Self {
            m: mm(self.m, tr(dcmarg(py, r)?)),
        }
        .validate()
    }
    #[pyo3(signature=(dtype=None))]
    fn __array__(&self, py: Python<'_>, dtype: Option<&PyAny>) -> PyResult<PyObject> {
        let a = pymat(py, &self.m);
        let np = py.import("numpy")?;
        let out = if let Some(d) = dtype {
            np.call_method1("asarray", (a, d))?
        } else {
            np.call_method1("asarray", (a,))?
        };
        Ok(out.call_method0("copy")?.into())
    }
}
impl DCM {
    fn validate(self) -> PyResult<Self> {
        if proper_dcm(self.m) {
            Ok(self)
        } else {
            Err(PyValueError::new_err(
                "DCM must be a proper rotation matrix",
            ))
        }
    }
}
