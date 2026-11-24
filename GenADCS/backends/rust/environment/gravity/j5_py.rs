#[pyclass(name = "J5", extends = JGravityModelPy)]
struct J5Py;

#[pymethods]
impl J5Py {
    #[classattr]
    fn degree() -> i32 { J5::DEGREE as i32 }

    #[new]
    fn new() -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(JGravityModelPy::configured(vec![J5::DEGREE as i32], None)?).add_subclass(Self))
    }
}
