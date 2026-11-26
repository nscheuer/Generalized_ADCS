/// Validated timestep configuration for an orbital solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitalSolverConfig {
    pub dt: f64,
}

impl OrbitalSolverConfig {
    pub fn new(dt: f64) -> Result<Self, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive seconds".to_owned());
        }
        Ok(Self { dt })
    }
}

#[cfg(test)]
mod tests {
    use super::OrbitalSolverConfig;

    #[test]
    fn validates_timestep() {
        assert_eq!(OrbitalSolverConfig::new(0.5).unwrap().dt, 0.5);
        assert!(OrbitalSolverConfig::new(0.0).is_err());
        assert!(OrbitalSolverConfig::new(f64::NAN).is_err());
    }
}
