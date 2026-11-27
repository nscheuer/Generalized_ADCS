use super::controller::{
    allocate_dipoles, read_first_sensor_body, Controller, SensorConfig, CTRL_BCROSS,
};

pub fn step(
    gain: f64,
    readings: &[f64],
    sensor_config: &SensorConfig,
    mtq_axes: &[[f64; 3]],
    mtq_max_dipole: &[f64],
    dipole_commands: &mut [f64],
) {
    let (ok_mag, b_body) = read_first_sensor_body(1, readings, sensor_config);
    let (ok_gyro, w_body) = read_first_sensor_body(0, readings, sensor_config);
    if !(ok_mag && ok_gyro) {
        dipole_commands.fill(0.0);
        return;
    }
    let b_sq = b_body[0] * b_body[0] + b_body[1] * b_body[1] + b_body[2] * b_body[2];
    if b_sq <= 1e-18 {
        dipole_commands.fill(0.0);
        return;
    }
    let scale = gain / b_sq;
    let dipole_body = [
        scale * (w_body[1] * b_body[2] - w_body[2] * b_body[1]),
        scale * (w_body[2] * b_body[0] - w_body[0] * b_body[2]),
        scale * (w_body[0] * b_body[1] - w_body[1] * b_body[0]),
    ];
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BCross {
    pub gain: f64,
}

impl BCross {
    pub fn new(gain: f64) -> Result<Self, String> {
        if !gain.is_finite() || gain <= 0.0 {
            return Err("BCross gain must be finite and positive".into());
        }
        Ok(Self { gain })
    }
}

impl Controller for BCross {
    fn kind(&self) -> i32 {
        CTRL_BCROSS
    }

    fn validate_sensors(&self, sensor_types: &[&str]) -> Result<(), String> {
        let has_mag = sensor_types.contains(&"Magnetometer");
        let has_gyro = sensor_types.contains(&"Gyroscope");
        if !has_mag || !has_gyro {
            return Err(
                "BCross controller requires both a Magnetometer and a Gyroscope onboard the satellite"
                    .into(),
            );
        }
        Ok(())
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.gain]
    }
}
