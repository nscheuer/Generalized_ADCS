// Rust-native magnetorquer configuration and validation.

/// Native magnetorquer geometry and dipole limit.
#[derive(Clone, Debug, PartialEq)]
pub struct MagnetorquerConfig {
    pub position: [f64; 3],
    pub orientation: [f64; 3],
    pub max_dipole: f64,
    pub misalignment: [f64; 3],
}

impl MagnetorquerConfig {
    /// Construct a magnetorquer after validating all finite values and limits.
    pub fn new(
        position: [f64; 3],
        orientation: [f64; 3],
        max_dipole: f64,
        misalignment: [f64; 3],
    ) -> Result<Self, &'static str> {
        if !position
            .iter()
            .chain(orientation.iter())
            .chain(misalignment.iter())
            .all(|v| v.is_finite())
            || !max_dipole.is_finite()
        {
            return Err("magnetorquer values must be finite");
        }
        if max_dipole < 0. {
            return Err("max_dipole must be non-negative");
        }
        if orientation.iter().all(|v| *v == 0.) {
            return Err("orientation must be non-zero");
        }
        Ok(Self {
            position,
            orientation,
            max_dipole,
            misalignment,
        })
    }
}
