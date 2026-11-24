#[derive(Clone, Debug, Default, PartialEq)]
pub struct InternalState {
    pub wheel_momentum: Vec<f64>,
}

impl InternalState {
    pub fn new(wheel_momentum: Vec<f64>) -> Result<Self, &'static str> {
        if wheel_momentum.iter().any(|value| !value.is_finite()) {
            return Err("wheel_momentum must contain only finite values");
        }
        Ok(Self { wheel_momentum })
    }
}
