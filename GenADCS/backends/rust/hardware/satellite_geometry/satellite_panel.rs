/// Rust-native surface panel used by satellite force and torque models.
#[derive(Clone, Debug, PartialEq)]
pub struct SatellitePanel {
    pub position: [f64; 3],
    pub orientation: [f64; 3],
    pub dimensions: [f64; 2],
    pub eta_s: f64,
    pub eta_d: f64,
    pub eta_a: f64,
    pub cd: f64,
}

impl SatellitePanel {
    pub fn new(position: [f64; 3], orientation: [f64; 3], dimensions: [f64; 2]) -> Self {
        Self {
            position,
            orientation,
            dimensions,
            eta_s: 0.0,
            eta_d: 0.0,
            eta_a: 1.0,
            cd: 2.2,
        }
    }
}
