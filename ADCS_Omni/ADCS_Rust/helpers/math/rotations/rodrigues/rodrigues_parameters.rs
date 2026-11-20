#[pyclass(name = "Rodriguez_Parameters")]
struct Rodrigues {
    p: [f64; 3],
}
#[pymethods]
impl Rodrigues {
    #[new]
    #[pyo3(signature=(values=None))]
    fn new(py: Python<'_>, values: Option<&PyAny>) -> PyResult<Self> {
        Ok(Self {
            p: if let Some(v) = values {
                fixed::<3>(py, v, "rotation parameters")?
            } else {
                [0.; 3]
            },
        })
    }
    #[staticmethod]
    fn identity() -> Self {
        Self { p: [0.; 3] }
    }
    #[staticmethod]
    fn from_quaternion(py: Python<'_>, q: &PyAny) -> PyResult<Self> {
        let a = qnormalize(quatarg(py, q)?)?;
        let mut q = a;
        if q[0] < 0. {
            q = q.map(|v| -v)
        }
        if q[0].abs() <= 1e-12 {
            return Err(PyValueError::new_err(
                "Rodrigues parameters are singular at a 180-degree rotation",
            ));
        }
        Self::new(
            py,
            Some(pyvec(py, [q[1] / q[0], q[2] / q[0], q[3] / q[0]])?.as_ref(py)),
        )
    }
    #[staticmethod]
    fn from_quaternion_array(py: Python<'_>, q: &PyAny) -> PyResult<Self> {
        Self::from_quaternion(py, q)
    }
    #[staticmethod]
    fn from_axis_angle(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<Self> {
        let q = _axis_angle_to_quaternion(py, axis, angle)?;
        Self::from_quaternion(py, q.as_ref(py))
    }
    #[staticmethod]
    fn from_rotation_vector(py: Python<'_>, v: &PyAny) -> PyResult<Self> {
        let q = rotation_vector_to_quaternion(py, v)?;
        Self::from_quaternion(py, q.as_ref(py))
    }
    #[staticmethod]
    fn from_dcm(py: Python<'_>, m: &PyAny) -> PyResult<Self> {
        let q = _dcm_to_quaternion(py, m)?;
        Self::from_quaternion(py, q.as_ref(py))
    }
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(py, self.p)
    }
    #[getter]
    fn hamilton_components(&self, py: Python<'_>) -> PyResult<PyObject> {
        rodrigues_to_quaternion(py, pyvec(py, self.p)?.as_ref(py))
    }
    #[getter]
    fn norm(&self) -> f64 {
        dot(&self.p, &self.p).sqrt()
    }
    fn as_vector(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(py, self.p)
    }
    fn to_quaternion(&self, py: Python<'_>) -> PyResult<Quaternion> {
        let q = rodrigues_to_quaternion(py, pyvec(py, self.p)?.as_ref(py))?;
        Quaternion::new(py, Some(q.as_ref(py)), "Hamilton")
    }
    fn to_dcm(&self, py: Python<'_>) -> PyResult<DCM> {
        let q = self.to_quaternion(py)?;
        let q = q.into_py(py);
        DCM::from_quaternion(py, q.as_ref(py))
    }
    fn rotate(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        self.to_quaternion(py)?.rotate(py, v)
    }
    fn inertial_to_body(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        self.rotate(py, v)
    }
    fn body_to_inertial(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        self.to_quaternion(py)?.body_to_inertial(py, v)
    }
    fn normalized(&self) -> Self {
        Self { p: self.p }
    }
    fn conjugate(&self, py: Python<'_>) -> PyResult<Self> {
        let q = self.to_quaternion(py)?.conjugate().into_py(py);
        Self::from_quaternion(py, q.as_ref(py))
    }
    fn inverse(&self, py: Python<'_>) -> PyResult<Self> {
        self.conjugate(py)
    }
    fn multiply(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        let a = self.to_quaternion(py)?;
        let b = Quaternion::new(
            py,
            Some(pyvec(py, quatarg(py, other)?)?.as_ref(py)),
            "Hamilton",
        )?;
        let q = a.multiply(py, b.into_py(py).as_ref(py))?.into_py(py);
        Self::from_quaternion(py, q.as_ref(py))
    }
    fn relative_to(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        let a = self.to_quaternion(py)?;
        let b = Quaternion::new(
            py,
            Some(pyvec(py, quatarg(py, other)?)?.as_ref(py)),
            "Hamilton",
        )?;
        let q = a.relative_to(py, b.into_py(py).as_ref(py))?.into_py(py);
        Self::from_quaternion(py, q.as_ref(py))
    }
    fn __mul__(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        self.multiply(py, other)
    }
    #[pyo3(signature=(dtype=None))]
    fn __array__(&self, py: Python<'_>, dtype: Option<&PyAny>) -> PyResult<PyObject> {
        let a = pyvec(py, self.p)?;
        let n = py.import("numpy")?;
        let out = if let Some(d) = dtype {
            n.call_method1("asarray", (a, d))?
        } else {
            n.call_method1("asarray", (a,))?
        };
        Ok(out.call_method0("copy")?.into())
    }
}
