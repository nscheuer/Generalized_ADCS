use crate::satellite_panel::SatellitePanel;

/// Rust-native physical properties of a satellite.
#[derive(Clone, Debug, PartialEq)]
pub struct SatelliteGeometry {
    pub mass: f64,
    pub inertia: [[f64; 3]; 3],
    pub panels: Vec<SatellitePanel>,
    pub center_of_mass: [f64; 3],
}

impl SatelliteGeometry {
    pub fn new(mass: f64, inertia: [[f64; 3]; 3]) -> Self {
        Self {
            mass,
            inertia,
            panels: Vec::new(),
            center_of_mass: [0.0; 3],
        }
    }
}
