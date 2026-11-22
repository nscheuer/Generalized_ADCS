#[pyclass(extends = JGravityModel)]
struct J2;

#[pymethods]
impl J2 {
    #[classattr]
    fn degree() -> i32 {
        2
    }

    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(JGravityModel::configured(vec![2], None).unwrap()).add_subclass(Self)
    }
}
