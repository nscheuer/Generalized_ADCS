/// Additive bias that grows linearly with elapsed time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearDrift {
    pub rate: f64,
    pub initial_value: f64,
}

impl LinearDrift {
    pub fn new(rate: f64, initial_value: f64) -> Result<Self, &'static str> {
        if !rate.is_finite() || !initial_value.is_finite() {
            return Err("linear drift values must be finite");
        }
        Ok(Self {
            rate,
            initial_value,
        })
    }

    pub fn apply(&self, values: &[f64], elapsed_seconds: f64) -> Result<Vec<f64>, &'static str> {
        if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err("elapsed time must be finite and non-negative");
        }
        let bias = self.initial_value + self.rate * elapsed_seconds;
        if !bias.is_finite() {
            return Err("linear drift result must be finite");
        }
        Ok(values.iter().map(|value| value + bias).collect())
    }
}
