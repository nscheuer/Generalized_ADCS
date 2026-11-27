/// Rust-native reaction wheel configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct ReactionWheelData {
    pub position: Vec<f64>,
    pub orientation: Vec<f64>,
    pub rotational_inertia: f64,
    pub max_torque: f64,
    pub max_momentum: f64,
    pub misalignment: Vec<f64>,
    pub noise: f64,
    pub bias: f64,
}

impl ReactionWheelData {
    pub fn new(
        position: Vec<f64>,
        orientation: Vec<f64>,
        rotational_inertia: f64,
        max_torque: f64,
        max_momentum: f64,
        misalignment: Vec<f64>,
        noise: f64,
        bias: f64,
    ) -> Result<Self, &'static str> {
        if position
            .iter()
            .chain(orientation.iter())
            .chain(misalignment.iter())
            .chain([rotational_inertia, max_torque, max_momentum, noise, bias].iter())
            .any(|value| !value.is_finite())
        {
            return Err("reaction wheel values must be finite");
        }
        if noise < 0.0 {
            return Err("noise must be nonnegative");
        }
        Ok(Self {
            position,
            orientation,
            rotational_inertia,
            max_torque,
            max_momentum,
            misalignment,
            noise,
            bias,
        })
    }
}
