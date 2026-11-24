/// Multiplicative scale-factor error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleFactorError {
    pub fraction: f64,
}

impl ScaleFactorError {
    pub fn new(fraction: f64) -> Result<Self, &'static str> {
        if !fraction.is_finite() {
            return Err("scale-factor fraction must be finite");
        }
        Ok(Self { fraction })
    }

    pub fn apply(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|value| value * (1.0 + self.fraction))
            .collect()
    }
}
