/// Configuration for additive uniform noise on a closed interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformNoise {
    pub lower: f64,
    pub upper: f64,
}

impl UniformNoise {
    pub fn new(lower: f64, upper: f64) -> Result<Self, &'static str> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err("uniform noise bounds must be finite");
        }
        if lower > upper {
            return Err("lower must be less than or equal to upper");
        }
        Ok(Self { lower, upper })
    }
}
