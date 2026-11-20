#[pyclass(extends = JGravityModel)]
struct J6;

#[pymethods]
impl J6 {
    #[classattr]
    fn degree() -> i32 {
        6
    }

    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(JGravityModel::configured(vec![6], None).unwrap()).add_subclass(Self)
    }
}
