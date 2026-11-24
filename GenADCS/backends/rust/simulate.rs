use std::ops::Range;

/// A validated sampled trajectory in seconds and state-vector rows.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub state_width: usize,
}

impl SimulationTrajectory {
    /// Validate a trajectory returned by a propagator.
    pub fn new(times: Vec<f64>, states: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if times.is_empty() || states.is_empty() || times.len() != states.len() {
            return Err("trajectory must contain matching non-empty times and states");
        }
        let state_width = states[0].len();
        if state_width < 13 || states.iter().any(|state| state.len() != state_width) {
            return Err("state vectors must have a consistent width of at least 13");
        }
        if times.iter().any(|time| !time.is_finite()) {
            return Err("trajectory times must be finite");
        }
        Ok(Self {
            times,
            states,
            state_width,
        })
    }

    /// Select the rows that belong in the requested fixed sampling interval.
    pub fn sampled_range(&self, interval: f64) -> Result<Range<usize>, &'static str> {
        if !interval.is_finite() || interval <= 0.0 {
            return Err("sampling interval must be finite and positive");
        }
        let last_time = *self.times.last().expect("validated non-empty times");
        let on_grid = (last_time / interval - (last_time / interval).round()).abs() < 1e-9;
        let end = if on_grid {
            self.times.len()
        } else {
            self.times.len() - 1
        };
        Ok(0..end)
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationTrajectory;

    fn state() -> Vec<f64> {
        vec![0.0; 13]
    }

    #[test]
    fn validates_dimensions_and_selects_aligned_samples() {
        let trajectory =
            SimulationTrajectory::new(vec![0.0, 1.0, 2.0], vec![state(), state(), state()])
                .unwrap();

        assert_eq!(trajectory.state_width, 13);
        assert_eq!(trajectory.sampled_range(1.0).unwrap(), 0..3);
    }

    #[test]
    fn excludes_final_off_grid_sample_and_rejects_bad_shapes() {
        let trajectory =
            SimulationTrajectory::new(vec![0.0, 1.0, 2.5], vec![state(), state(), state()])
                .unwrap();

        assert_eq!(trajectory.sampled_range(1.0).unwrap(), 0..2);
        assert!(SimulationTrajectory::new(vec![0.0], vec![vec![0.0; 12]]).is_err());
        assert!(SimulationTrajectory::new(vec![0.0, 1.0], vec![state()]).is_err());
        assert!(trajectory.sampled_range(0.0).is_err());
    }
}
