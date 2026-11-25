/// Configuration for first-order Gauss-Markov noise.
///
/// This mirrors the Python dataclass: it stores the standard deviation and
/// correlation time, while time evolution is performed by simulation code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussMarkovNoise {
    pub standard_deviation: f64,
    pub correlation_time: f64,
}

impl GaussMarkovNoise {
    pub fn new(standard_deviation: f64, correlation_time: f64) -> Self {
        Self {
            standard_deviation,
            correlation_time,
        }
    }
}
