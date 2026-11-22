#[pyclass]
struct Quaternion {
    q: [f64; 4],
    convention: String,
}
#[pymethods]
impl Quaternion {
    #[new]
    #[pyo3(signature=(values=None, convention="Hamilton"))]
    fn new(py: Python<'_>, values: Option<&PyAny>, convention: &str) -> PyResult<Self> {
        let name = convention.to_lowercase();
        let input = match values {
            Some(x) => fixed::<4>(py, x, "quaternion")?,
            None => [1., 0., 0., 0.],
        };
        let (q, stored) = match name.as_str() {
            "hamilton" | "h" => (qnormalize(input)?, "Hamilton"),
            "jpl" | "j" => (
                qnormalize([input[3], -input[0], -input[1], -input[2]])?,
                "JPL",
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "convention must be 'Hamilton' or 'JPL'",
                ))
            }
        };
        Ok(Self {
            q,
            convention: stored.into(),
        })
    }
    #[staticmethod]
    fn identity() -> Self {
        Self {
            q: [1., 0., 0., 0.],
            convention: "Hamilton".into(),
        }
    }
    #[staticmethod]
    fn from_axis_angle(py: Python<'_>, axis: &PyAny, angle: f64) -> PyResult<Self> {
        let q: _ = _axis_angle_to_quaternion(py, axis, angle)?;
        Self::new(py, Some(q.as_ref(py)), "Hamilton")
    }
    #[staticmethod]
    fn from_rotation_vector(py: Python<'_>, v: &PyAny) -> PyResult<Self> {
        let q = rotation_vector_to_quaternion(py, v)?;
        Self::new(py, Some(q.as_ref(py)), "Hamilton")
    }
    #[staticmethod]
    fn from_dcm(py: Python<'_>, m: &PyAny) -> PyResult<Self> {
        let a = mat3(py, m, "matrix")?;
        if !proper_dcm(a) {
            return Err(PyValueError::new_err(
                "matrix must be a proper rotation matrix",
            ));
        }
        let q = _dcm_to_quaternion(py, m)?;
        Self::new(py, Some(q.as_ref(py)), "Hamilton")
    }
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(
            py,
            if self.convention == "JPL" {
                [-self.q[1], -self.q[2], -self.q[3], self.q[0]]
            } else {
                self.q
            },
        )
    }
    #[getter]
    fn hamilton_components(&self, py: Python<'_>) -> PyResult<PyObject> {
        pyvec(py, self.q)
    }
    #[getter]
    fn convention(&self) -> String {
        self.convention.clone()
    }
    #[getter]
    fn norm(&self) -> f64 {
        qnorm(self.q)
    }
    #[pyo3(signature=(convention="Hamilton"))]
    fn as_convention(&self, py: Python<'_>, convention: &str) -> PyResult<PyObject> {
        let k = convention.to_lowercase();
        match k.as_str() {
            "hamilton" | "h" => pyvec(py, self.q),
            "jpl" | "j" => pyvec(py, [-self.q[1], -self.q[2], -self.q[3], self.q[0]]),
            _ => Err(PyValueError::new_err(
                "convention must be 'Hamilton' or 'JPL'",
            )),
        }
    }
    #[pyo3(signature=(convention="Hamilton"))]
    fn to_convention(&self, py: Python<'_>, convention: &str) -> PyResult<Self> {
        let v = self.as_convention(py, convention)?;
        Self::new(py, Some(v.as_ref(py)), convention)
    }
    fn normalized(&self) -> Self {
        Self {
            q: self.q,
            convention: "Hamilton".into(),
        }
    }
    fn conjugate(&self) -> Self {
        Self {
            q: qconj(self.q),
            convention: "Hamilton".into(),
        }
    }
    fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            q: qinv(self.q)?,
            convention: "Hamilton".into(),
        })
    }
    fn multiply(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(Self {
            q: qmul(self.q, quatarg(py, other)?),
            convention: "Hamilton".into(),
        })
    }
    fn __mul__(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        self.multiply(py, other)
    }
    fn rotate(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        pyvec(py, qrotate(self.q, fixed(py, v, "vector")?))
    }
    fn inertial_to_body(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        self.rotate(py, v)
    }
    fn body_to_inertial(&self, py: Python<'_>, v: &PyAny) -> PyResult<PyObject> {
        pyvec(py, qrotate(qconj(self.q), fixed(py, v, "vector")?))
    }
    fn to_dcm(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(pymat(py, &qdcm(self.q)))
    }
    fn relative_to(&self, py: Python<'_>, reference: &PyAny) -> PyResult<Self> {
        let r = quatarg(py, reference)?;
        let mut q = qnormalize(qmul(self.q, qinv(r)?))?;
        if q[0] < 0. {
            q = q.map(|v| -v)
        }
        Ok(Self {
            q,
            convention: "Hamilton".into(),
        })
    }
    #[pyo3(signature=(dtype=None))]
    fn __array__(&self, py: Python<'_>, dtype: Option<&PyAny>) -> PyResult<PyObject> {
        let a = pyvec(py, self.q)?;
        let np = py.import("numpy")?;
        let out = if let Some(d) = dtype {
            np.call_method1("asarray", (a, d))?
        } else {
            np.call_method1("asarray", (a,))?
        };
        Ok(out.call_method0("copy")?.into())
    }
    fn __repr__(&self) -> String {
        format!(
            "Quaternion({:?}, convention='Hamilton', frame='q_B<-I')",
            self.q
        )
    }
}
