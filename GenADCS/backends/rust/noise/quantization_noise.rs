/// Configuration for bounded-step quantization noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizationNoise {
    pub step_size: f64,
}

impl QuantizationNoise {
    pub fn new(step_size: f64) -> Result<Self, &'static str> {
        if !step_size.is_finite() || step_size < 0.0 {
            return Err("step_size must be finite and non-negative");
        }
        Ok(Self { step_size })
    }
}
