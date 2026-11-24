#[pyclass(name = "J2", extends = JGravityModelPy)]
struct J2Py;

#[pymethods]
impl J2Py {
    #[classattr]
    fn degree() -> i32 { J2::DEGREE as i32 }

    #[new]
    fn new() -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(JGravityModelPy::configured(vec![J2::DEGREE as i32], None)?).add_subclass(Self))
    }
}
