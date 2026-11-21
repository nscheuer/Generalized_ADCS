#[pyclass(extends=PlanetaryBody)]
pub struct Earth;

#[pymethods]
impl Earth {
    #[new]
    fn new() -> (Self, PlanetaryBody) {
        (
            Self,
            PlanetaryBody {
                name: "Earth".to_owned(),
                gravitational_parameter: 3.986004418e14,
                reference_radius: 6_378_137.0,
            },
        )
    }
}
