use math_native::quaternion_to_dcm_native;

pub use crate::estimator::attitude_estimator::attitude_mekf::SensorConfig;

pub const CTRL_NONE: i32 = 0;
pub const CTRL_BDOT: i32 = 1;
pub const CTRL_BCROSS: i32 = 2;
pub const CTRL_ATTITUDE: i32 = 3;

pub fn read_first_sensor_body(
    target_code: i64,
    readings: &[f64],
    sensor_config: &SensorConfig,
) -> (bool, [f64; 3]) {
    for i in 0..sensor_config.code.len() {
        if sensor_config.code[i] == target_code && sensor_config.enabled[i] {
            let off = sensor_config.offsets[i];
            if off + 3 <= readings.len() {
                let raw = [readings[off], readings[off + 1], readings[off + 2]];
                if !raw[0].is_nan() && !raw[1].is_nan() && !raw[2].is_nan() {
                    let dcm = quaternion_to_dcm_native(sensor_config.orientation[i]);
                    let dcm_t = dcm.transpose();
                    let body = [
                        dcm_t[(0, 0)] * raw[0] + dcm_t[(0, 1)] * raw[1] + dcm_t[(0, 2)] * raw[2],
                        dcm_t[(1, 0)] * raw[0] + dcm_t[(1, 1)] * raw[1] + dcm_t[(1, 2)] * raw[2],
                        dcm_t[(2, 0)] * raw[0] + dcm_t[(2, 1)] * raw[1] + dcm_t[(2, 2)] * raw[2],
                    ];
                    return (true, body);
                }
            }
        }
    }
    (false, [0.0; 3])
}

pub fn allocate_dipoles(
    dipole_body: [f64; 3],
    mtq_axes: &[[f64; 3]],
    mtq_max_dipole: &[f64],
    dipole_commands: &mut [f64],
) {
    for i in 0..dipole_commands.len() {
        let mut cmd = dipole_body[0] * mtq_axes[i][0]
            + dipole_body[1] * mtq_axes[i][1]
            + dipole_body[2] * mtq_axes[i][2];
        let limit = mtq_max_dipole[i];
        if cmd > limit {
            cmd = limit;
        } else if cmd < -limit {
            cmd = -limit;
        }
        dipole_commands[i] = cmd;
    }
}

pub trait Controller: Send + Sync {
    fn kind(&self) -> i32;
    fn validate_sensors(&self, sensor_types: &[&str]) -> Result<(), String>;
    fn to_params(&self) -> Vec<f64>;
    fn feedback_matrices(&self) -> Option<(super::feedback::Mat3, super::feedback::Mat3)> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoController;

impl Controller for NoController {
    fn kind(&self) -> i32 {
        CTRL_NONE
    }

    fn validate_sensors(&self, _sensor_types: &[&str]) -> Result<(), String> {
        Ok(())
    }

    fn to_params(&self) -> Vec<f64> {
        vec![0.0]
    }
}
