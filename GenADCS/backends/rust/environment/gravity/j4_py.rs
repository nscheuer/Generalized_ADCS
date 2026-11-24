#[pyclass(name = "J4", extends = JGravityModelPy)]
struct J4Py;

#[pymethods]
impl J4Py {
    #[classattr]
    fn degree() -> i32 { J4::DEGREE as i32 }

    #[new]
    fn new() -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(JGravityModelPy::configured(vec![J4::DEGREE as i32], None)?).add_subclass(Self))
    }
}
