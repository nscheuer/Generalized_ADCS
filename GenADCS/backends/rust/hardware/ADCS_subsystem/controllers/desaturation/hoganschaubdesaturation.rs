use super::super::controller::{allocate_dipoles, read_first_sensor_body, SensorConfig};
use super::desaturation::{Desaturation, DESAT_HOGAN_SCHAUB};

pub fn hogan_schaub_u0(wheel_momenta: &[f64], gain: f64, target_momentum: f64) -> Vec<f64> {
    wheel_momenta
        .iter()
        .map(|&h| gain * (h - target_momentum))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn hogan_schaub_mtq_dump(
    wheel_momenta: &[f64],
    wheel_axes: &[[f64; 3]],
    mtq_gain: f64,
    target_momentum: f64,
    readings: &[f64],
    sensor_config: &SensorConfig,
    mtq_axes: &[[f64; 3]],
    mtq_max_dipole: &[f64],
    dipole_commands: &mut [f64],
) {
    if mtq_gain <= 0.0 || dipole_commands.is_empty() {
        dipole_commands.fill(0.0);
        return;
    }
    let (ok_mag, b_body) = read_first_sensor_body(1, readings, sensor_config);
    if !ok_mag {
        dipole_commands.fill(0.0);
        return;
    }
    let b_sq = b_body[0] * b_body[0] + b_body[1] * b_body[1] + b_body[2] * b_body[2];
    if b_sq <= 1e-18 {
        dipole_commands.fill(0.0);
        return;
    }
    let mut h_err_body = [0.0; 3];
    for (w, &h) in wheel_momenta.iter().enumerate() {
        let diff = h - target_momentum;
        h_err_body[0] += wheel_axes[w][0] * diff;
        h_err_body[1] += wheel_axes[w][1] * diff;
        h_err_body[2] += wheel_axes[w][2] * diff;
    }
    let scale = mtq_gain / b_sq;
    let dipole_body = [
        scale * (h_err_body[1] * b_body[2] - h_err_body[2] * b_body[1]),
        scale * (h_err_body[2] * b_body[0] - h_err_body[0] * b_body[2]),
        scale * (h_err_body[0] * b_body[1] - h_err_body[1] * b_body[0]),
    ];
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoganSchaubManagement {
    pub gain: f64,
    pub target_momentum: f64,
    pub mtq_gain: f64,
}

impl HoganSchaubManagement {
    pub fn new(gain: f64, target_momentum: f64, mtq_gain: f64) -> Result<Self, String> {
        if !gain.is_finite() || gain <= 0.0 {
            return Err("HoganSchaubManagement gain must be finite and positive".into());
        }
        if !target_momentum.is_finite() {
            return Err("target_momentum must be finite".into());
        }
        if !mtq_gain.is_finite() || mtq_gain < 0.0 {
            return Err("mtq_gain must be finite and nonnegative".into());
        }
        Ok(Self {
            gain,
            target_momentum,
            mtq_gain,
        })
    }
}

impl Desaturation for HoganSchaubManagement {
    fn kind(&self) -> i32 {
        DESAT_HOGAN_SCHAUB
    }

    fn to_params(&self) -> Vec<f64> {
        vec![self.gain, self.target_momentum, self.mtq_gain]
    }
}
