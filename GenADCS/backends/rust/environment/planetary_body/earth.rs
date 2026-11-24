use super::planetary_body::PlanetaryBody;

/// Earth's standard gravitational parameter and mean equatorial radius.
pub struct Earth;

impl Earth {
    pub const GRAVITATIONAL_PARAMETER: f64 = 3.986004418e14;
    pub const REFERENCE_RADIUS: f64 = 6_378_137.0;

    pub fn planetary_body() -> PlanetaryBody {
        PlanetaryBody::new(
            "Earth",
            Self::GRAVITATIONAL_PARAMETER,
            Self::REFERENCE_RADIUS,
        )
        .expect("Earth constants satisfy planetary body validation")
    }
}
