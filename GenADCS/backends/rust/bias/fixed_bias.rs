/// Constant additive bias applied to a measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedBias {
    pub value: f64,
}

impl FixedBias {
    pub fn new(value: f64) -> Result<Self, &'static str> {
        if !value.is_finite() {
            return Err("fixed bias must be finite");
        }
        Ok(Self { value })
    }

    pub fn apply(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|value| value + self.value).collect()
    }
}
