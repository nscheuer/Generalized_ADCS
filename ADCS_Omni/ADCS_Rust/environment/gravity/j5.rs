#[pyclass(extends = JGravityModel)]
struct J5;

#[pymethods]
impl J5 {
    #[classattr]
    fn degree() -> i32 {
        5
    }

    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(JGravityModel::configured(vec![5], None).unwrap()).add_subclass(Self)
    }
}
