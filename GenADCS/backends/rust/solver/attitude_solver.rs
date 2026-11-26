/// Native attitude solver configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttitudeSolverConfig {
    pub dt: f64,
}

impl AttitudeSolverConfig {
    pub fn new(dt: f64) -> Result<Self, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive seconds".to_owned());
        }
        Ok(Self { dt })
    }
}

#[cfg(test)]
mod tests {
    use super::AttitudeSolverConfig;

    #[test]
    fn validates_positive_finite_step() {
        assert_eq!(AttitudeSolverConfig::new(0.25).unwrap().dt, 0.25);
        assert!(AttitudeSolverConfig::new(0.0).is_err());
        assert!(AttitudeSolverConfig::new(f64::NAN).is_err());
    }
}
