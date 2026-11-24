#[pyclass(name = "J3", extends = JGravityModelPy)]
struct J3Py;

#[pymethods]
impl J3Py {
    #[classattr]
    fn degree() -> i32 { J3::DEGREE as i32 }

    #[new]
    fn new() -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(JGravityModelPy::configured(vec![J3::DEGREE as i32], None)?).add_subclass(Self))
    }
}
