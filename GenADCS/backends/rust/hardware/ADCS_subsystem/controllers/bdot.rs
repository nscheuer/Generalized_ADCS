use super::controller::{
    allocate_dipoles, read_first_sensor_body, Controller, SensorConfig, CTRL_BDOT,
};

#[allow(clippy::too_many_arguments)]
pub fn step(
    gain: f64,
    dt_ctrl: f64,
    readings: &[f64],
    sensor_config: &SensorConfig,
    mtq_axes: &[[f64; 3]],
    mtq_max_dipole: &[f64],
    ctrl_work: &mut [f64],
    dipole_commands: &mut [f64],
) {
    let (ok_mag, b_body) = read_first_sensor_body(1, readings, sensor_config);
    if !ok_mag {
        dipole_commands.fill(0.0);
        return;
    }
    if ctrl_work[3] == 0.0 || dt_ctrl <= 0.0 {
        ctrl_work[0] = b_body[0];
        ctrl_work[1] = b_body[1];
        ctrl_work[2] = b_body[2];
        ctrl_work[3] = 1.0;
        dipole_commands.fill(0.0);
        return;
    }
    let b_dot = [
        (b_body[0] - ctrl_work[0]) / dt_ctrl,
        (b_body[1] - ctrl_work[1]) / dt_ctrl,
        (b_body[2] - ctrl_work[2]) / dt_ctrl,
    ];
    ctrl_work[0] = b_body[0];
    ctrl_work[1] = b_body[1];
    ctrl_work[2] = b_body[2];
    let dipole_body = [-gain * b_dot[0], -gain * b_dot[1], -gain * b_dot[2]];
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BDot {
    pub gain: f64,
}

impl BDot {
    pub fn new(gain: f64) -> Result<Self, String> {
        if !gain.is_finite() || gain <= 0.0 {
            return Err("BDot gain must be finite and positive".into());
        }
        Ok(Self { gain })
    }
}

impl Controller for BDot {
    fn kind(&self) -> i32 {
        CTRL_BDOT
    }

    fn validate_sensors(&self, sensor_types: &[&str]) -> Result<(), String> {
        if !sensor_types.contains(&"Magnetometer") {
            return Err("BDot controller requires a Magnetometer onboard the satellite".into());
        }
        Ok(())
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.gain]
    }
}
