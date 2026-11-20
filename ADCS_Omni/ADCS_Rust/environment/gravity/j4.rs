#[pyclass(extends = JGravityModel)]
struct J4;

#[pymethods]
impl J4 {
    #[classattr]
    fn degree() -> i32 {
        4
    }

    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(JGravityModel::configured(vec![4], None).unwrap()).add_subclass(Self)
    }
}
