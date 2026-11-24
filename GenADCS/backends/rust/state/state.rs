use crate::attitude_state::AttitudeState;
use crate::internal_state::InternalState;
use crate::orbital_state::OrbitalState;

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub attitude: AttitudeState,
    pub orbit: OrbitalState,
    pub internal: InternalState,
}

impl State {
    pub fn new(attitude: AttitudeState, orbit: OrbitalState, internal: InternalState) -> Self {
        Self {
            attitude,
            orbit,
            internal,
        }
    }

    pub fn validate_wheel_count(&self, wheel_count: usize) -> Result<(), String> {
        if wheel_count != self.internal.wheel_momentum.len() {
            return Err(format!(
                "expected {wheel_count} reaction wheel momenta, got {}",
                self.internal.wheel_momentum.len()
            ));
        }
        Ok(())
    }

    pub fn normalized_quaternion(&self) -> Result<[f64; 4], &'static str> {
        self.attitude.normalized_quaternion()
    }

    pub fn cartesian_orbit(&self) -> Result<([f64; 3], [f64; 3]), &'static str> {
        self.orbit.cartesian()
    }
}
