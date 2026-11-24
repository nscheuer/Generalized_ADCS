#[pyclass(name = "J6", extends = JGravityModelPy)]
struct J6Py;

#[pymethods]
impl J6Py {
    #[classattr]
    fn degree() -> i32 { J6::DEGREE as i32 }

    #[new]
    fn new() -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(JGravityModelPy::configured(vec![J6::DEGREE as i32], None)?).add_subclass(Self))
    }
}
