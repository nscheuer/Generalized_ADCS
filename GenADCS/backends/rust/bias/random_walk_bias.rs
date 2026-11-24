/// Additive random-walk bias parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomWalkBias {
    pub standard_deviation: f64,
    pub initial_value: f64,
}

impl RandomWalkBias {
    pub fn new(standard_deviation: f64, initial_value: f64) -> Result<Self, &'static str> {
        if !standard_deviation.is_finite() || standard_deviation < 0.0 || !initial_value.is_finite()
        {
            return Err(
                "random-walk bias values must be finite and standard deviation non-negative",
            );
        }
        Ok(Self {
            standard_deviation,
            initial_value,
        })
    }

    pub fn apply(
        &self,
        values: &[f64],
        elapsed_seconds: f64,
        standard_normals: &[f64],
    ) -> Result<Vec<f64>, &'static str> {
        if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err("elapsed time must be finite and non-negative");
        }
        if values.len() != standard_normals.len()
            || standard_normals.iter().any(|value| !value.is_finite())
        {
            return Err("one finite standard-normal value is required per measurement");
        }
        let scale = self.standard_deviation * elapsed_seconds.sqrt();
        Ok(values
            .iter()
            .zip(standard_normals)
            .map(|(value, normal)| value + self.initial_value + scale * normal)
            .collect())
    }
}
