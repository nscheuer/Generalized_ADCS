#[pyclass(name = "Modified_Rodriguez_Parameters")]
struct MRP {
    p: [f64; 3],
}
#[pymethods]
impl MRP {
    #[new]
    #[pyo3(signature=(values=None))]
    fn new(py: Python<'_>, values: Option<&PyAny>) -> PyResult<Self> {
        let mut p = if let Some(v) = values {
            fixed::<3>(py, v, "rotation parameters")?
        } else {
            [0.; 3]
        };
        let n = dot(&p, &p);
        if n > 1.0 {
            p = p.map(|v| -v / n)
        }
        Ok(Self { p })
    }
    #[staticmethod]
    fn identity() -> Self {
        Self { p: [0.; 3] }
    }
    #[staticmethod]
    fn from_quaternion(py: Python<'_>, q: &PyAny) -> PyResult<Self> {
        let q = pyvec(py, quatarg(py, q)?)?;
        let p = quaternion_to_mrp(py, q.as_ref(py))?;
        Self::new(py, Some(p.as_ref(py)))
    }
    #[staticmethod]
    fn from_quaternion_array(py: Python<'_>, q: &PyAny) -> PyResult<Self> {
        let p = quaternion_to_mrp(py, q)?;
        Self::new(py, Some(p.as_ref(py)))
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
    fn switch_shadow_set(&self) -> PyResult<Self> {
        let n = dot(&self.p, &self.p);
        if n == 0. {
            return Err(PyValueError::new_err(
                "the identity MRP has no finite shadow set",
            ));
        }
        Ok(Self {
            p: self.p.map(|v| -v / n),
        })
    }
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(py, self.p)
    }
    #[getter]
    fn hamilton_components(&self, py: Python<'_>) -> PyResult<PyObject> {
        mrp_to_quaternion(py, pyvec(py, self.p)?.as_ref(py))
    }
    #[getter]
    fn norm(&self) -> f64 {
        dot(&self.p, &self.p).sqrt()
    }
    fn as_vector(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(py, self.p)
    }
    fn to_quaternion(&self, py: Python<'_>) -> PyResult<Quaternion> {
        Quaternion::new(
            py,
            Some(mrp_to_quaternion(py, pyvec(py, self.p)?.as_ref(py))?.as_ref(py)),
            "Hamilton",
        )
    }
    fn to_dcm(&self, py: Python<'_>) -> PyResult<DCM> {
        DCM::from_quaternion(py, self.to_quaternion(py)?.into_py(py).as_ref(py))
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
        Self::from_quaternion(
            py,
            self.to_quaternion(py)?.conjugate().into_py(py).as_ref(py),
        )
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
        Self::from_quaternion(
            py,
            a.multiply(py, b.into_py(py).as_ref(py))?
                .into_py(py)
                .as_ref(py),
        )
    }
    fn relative_to(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        let a = self.to_quaternion(py)?;
        let b = Quaternion::new(
            py,
            Some(pyvec(py, quatarg(py, other)?)?.as_ref(py)),
            "Hamilton",
        )?;
        Self::from_quaternion(
            py,
            a.relative_to(py, b.into_py(py).as_ref(py))?
                .into_py(py)
                .as_ref(py),
        )
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
