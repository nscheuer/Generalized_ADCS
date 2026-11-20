#[pyclass(extends = JGravityModel)]
struct J3;

#[pymethods]
impl J3 {
    #[classattr]
    fn degree() -> i32 {
        3
    }

    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(JGravityModel::configured(vec![3], None).unwrap()).add_subclass(Self)
    }
}
