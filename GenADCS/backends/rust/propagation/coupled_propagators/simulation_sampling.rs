//! Pure-Rust in-loop sensor sampling and attitude estimation.

use super::{sensor_kernels as k, Runtime};

pub const SENSOR_NAMES: [&str; 8] = [
    "gyroscope",
    "magnetometer",
    "autonomous_star_tracker",
    "coarse_sun_sensor",
    "fine_sun_sensor",
    "fine_earth_sensor",
    "gnss",
    "accelerometer",
];

pub const SENSOR_CLASSES: [&str; 8] = [
    "Gyroscope",
    "Magnetometer",
    "AutonomousStarTracker",
    "CoarseSunSensor",
    "FineSunSensor",
    "FineEarthSensor",
    "GNSS",
    "Accelerometer",
];

pub const SENSOR_WIDTH: [usize; 8] = [3, 3, 4, 1, 2, 2, 6, 3];

#[derive(Clone, Debug)]
pub struct SensorSpec {
    pub kind: usize,
    pub offset: usize,
    pub enabled: bool,
    pub ori: [f64; 4],
    pub mis: [f64; 4],
    pub bias: [f64; 6],
    pub chol: [[f64; 6]; 6],
    pub normal: [f64; 3],
    pub fov: [f64; 2],
    pub saturation: f64,
    pub sensitivity: f64,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub enum EstimateState {
    Mekf(super::base::AttitudeEstimator),
    Ekf(super::attitude_ekf::AttitudeEKF),
}

pub struct SimulationSampler {
    pub specs: Vec<SensorSpec>,
    pub counts: [usize; 8],
    pub estimator_config: super::attitude_mekf::SensorConfig,
    pub estimate: Option<EstimateState>,
    pub is_ekf: bool,
    pub epoch_jd: f64,
    pub seed: u64,
    pub nees: bool,
    pub previous_time: f64,
    pub sun_needed: bool,
    pub recorded_clean: [Vec<f64>; 8],
    pub recorded_reading: [Vec<f64>; 8],
    pub recorded_estimates: Vec<f64>,
    pub recorded_rows: usize,
    scratch_clean: [Vec<f64>; 8],
    scratch_reading: [Vec<f64>; 8],
    scratch_estimate: Vec<f64>,
    scratch_readings_flat: Vec<f64>,
}

impl SimulationSampler {
    pub fn new(
        specs: Vec<SensorSpec>,
        counts: [usize; 8],
        estimate: Option<EstimateState>,
        epoch_jd: f64,
        seed: u64,
        nees: bool,
    ) -> Self {
        let is_ekf = matches!(estimate, Some(EstimateState::Ekf(_)));
        let mut estimator_config = super::attitude_mekf::SensorConfig::default();
        let mut total_width = 0;
        for s in &specs {
            estimator_config.code.push(s.kind as i64);
            estimator_config.orientation.push(s.ori);
            estimator_config.misalignment.push(s.mis);
            estimator_config.bias.push(s.bias);
            estimator_config.noise_cholesky.push(s.chol);
            estimator_config.normal.push(s.normal);
            estimator_config.fov.push(s.fov);
            estimator_config
                .scalars
                .push([s.saturation, s.sensitivity, s.std]);
            estimator_config.reference.push([0.0; 3]);
            estimator_config.enabled.push(s.enabled);
            estimator_config.offsets.push(total_width);
            total_width += SENSOR_WIDTH[s.kind];
        }
        let sun_needed = counts[2] + counts[3] + counts[4] > 0;
        let scratch_clean = std::array::from_fn(|k| vec![f64::NAN; counts[k]]);
        let scratch_reading = std::array::from_fn(|k| vec![f64::NAN; counts[k]]);
        let est_width = if is_ekf { 109 } else { 96 };
        let scratch_estimate = if estimate.is_some() {
            vec![f64::NAN; est_width]
        } else {
            Vec::new()
        };
        let scratch_readings_flat = vec![f64::NAN; total_width];
        Self {
            specs,
            counts,
            estimator_config,
            estimate,
            is_ekf,
            epoch_jd,
            seed,
            nees,
            previous_time: 0.0,
            sun_needed,
            recorded_clean: std::array::from_fn(|_| Vec::new()),
            recorded_reading: std::array::from_fn(|_| Vec::new()),
            recorded_estimates: Vec::new(),
            recorded_rows: 0,
            scratch_clean,
            scratch_reading,
            scratch_estimate,
            scratch_readings_flat,
        }
    }

    pub fn sample(
        &mut self,
        y: &[f64],
        time: f64,
        tick: u64,
        r: &Runtime,
    ) -> Result<(), String> {
        for k in 0..8 {
            self.scratch_clean[k].fill(f64::NAN);
            self.scratch_reading[k].fill(f64::NAN);
        }
        let jd = self.epoch_jd + time / 86400.0;
        let q = [y[0], y[1], y[2], y[3]];
        let w = super::vec3(&y[4..]);
        let pos = super::vec3(&y[7..]);
        let vel = super::vec3(&y[10..]);
        let earth = pos.map(|x| -x);
        let mut sun = [0.0; 3];
        let mut visible = false;
        if self.sun_needed {
            let eph = r
                .ephemeris
                .as_ref()
                .ok_or_else(|| "Sun and star tracker sensors require an ephemeris".to_owned())?;
            let tdb = magnetic_native::native_tdb_jd(jd).map_err(|e| e.to_string())?;
            let sp = super::scale3(eph.position_km(0, tdb).map_err(|e| e.to_string())?, 1000.0);
            sun = super::sub3(sp, pos);
            visible = ephemeris_native::native_eclipse(pos, sp, r.radius, 696340000.0).0 > 0.0;
        }
        let field = if self.counts[1] > 0 {
            r.magnetic
                .as_ref()
                .ok_or_else(|| "magnetometer requires a magnetic model".to_owned())?
                .field_inertial_tesla(pos, jd)
                .map_err(|e| e.to_string())?
        } else {
            [0.0; 3]
        };
        let specific = if self.counts[7] > 0 {
            let d = super::derivative_vec(y, jd, r)?;
            let g = gravity_native::native_acceleration(pos, r.mu, r.radius, &r.zonals)
                .map_err(|e| e.to_string())?;
            super::sub3(super::vec3(&d[10..]), g)
        } else {
            [0.0; 3]
        };

        for (index, s) in self.specs.iter().enumerate() {
            if !s.enabled {
                continue;
            }
            let kind = s.kind;
            let ori = s.ori;
            let mis = s.mis;
            if kind == 3 && !visible {
                continue;
            }
            if (kind == 4 || kind == 5)
                && !k::fine_direction_available(
                    if kind == 4 { sun } else { earth },
                    q,
                    ori,
                    mis,
                    s.fov,
                    true,
                    kind == 5 || visible,
                )
            {
                continue;
            }
            if kind == 2
                && [earth, sun].iter().any(|&v| {
                    let v = k::rotate_inertial_to_sensor(v, q, ori, mis);
                    v[2] > 0.0 && v[0].hypot(v[1]).atan2(v[2]) < s.fov[0]
                })
            {
                continue;
            }
            let draw = philox_native::philox::standard_normals(self.seed, tick, index as u32, 6);
            let l = std::array::from_fn(|i| std::array::from_fn(|j| s.chol[i][j]));
            let b = [s.bias[0], s.bias[1], s.bias[2]];
            let z = Some([draw[0], draw[1], draw[2]]);
            for noisy in [false, true] {
                let dest = if noisy {
                    &mut self.scratch_reading[kind]
                } else {
                    &mut self.scratch_clean[kind]
                };
                match kind {
                    0 => {
                        let v = k::rate_measurement(
                            w, ori, mis, b, l, s.saturation, noisy, noisy, z,
                        );
                        dest[s.offset..s.offset + 3].copy_from_slice(&v);
                    }
                    1 | 7 => {
                        let v = k::vector_measurement(
                            if kind == 1 { field } else { specific },
                            q,
                            ori,
                            mis,
                            b,
                            l,
                            s.saturation,
                            noisy,
                            noisy,
                            z,
                        );
                        dest[s.offset..s.offset + 3].copy_from_slice(&v);
                    }
                    2 => {
                        let v = k::star_tracker_measurement(q, ori, mis, l, noisy, z);
                        dest[s.offset..s.offset + 4].copy_from_slice(&v);
                    }
                    3 => {
                        let v = k::scalar_sun_measurement(
                            sun,
                            q,
                            ori,
                            mis,
                            s.normal,
                            s.sensitivity,
                            s.bias[0],
                            s.std,
                            s.saturation,
                            noisy,
                            noisy,
                            Some(draw[0]),
                        );
                        dest[s.offset] = v;
                    }
                    4 | 5 => {
                        let v = k::fine_direction_measurement(
                            if kind == 4 { sun } else { earth },
                            q,
                            ori,
                            mis,
                            [s.bias[0], s.bias[1]],
                            [[s.chol[0][0], s.chol[0][1]], [s.chol[1][0], s.chol[1][1]]],
                            noisy,
                            noisy,
                            Some([draw[0], draw[1]]),
                        );
                        dest[s.offset..s.offset + 2].copy_from_slice(&v);
                    }
                    _ => {
                        for i in 0..6 {
                            let base = if i < 3 { pos[i] } else { vel[i - 3] };
                            dest[s.offset + i] = if noisy {
                                base + s.bias[i]
                                    + (0..6).map(|j| s.chol[i][j] * draw[j]).sum::<f64>()
                            } else {
                                base
                            };
                        }
                    }
                }
            }
        }

        if let Some(est) = self.estimate.as_mut() {
            self.scratch_readings_flat.fill(f64::NAN);
            for (i, s) in self.specs.iter().enumerate() {
                self.estimator_config.reference[i] = match s.kind {
                    1 => field,
                    3 | 4 => sun,
                    5 => earth,
                    _ => [0.0; 3],
                };
                let off = self.estimator_config.offsets[i];
                let w_len = SENSOR_WIDTH[s.kind];
                self.scratch_readings_flat[off..off + w_len]
                    .copy_from_slice(&self.scratch_reading[s.kind][s.offset..s.offset + w_len]);
            }
            let dt = time - self.previous_time;
            self.previous_time = time;
            match est {
                EstimateState::Mekf(mekf) => {
                    let values = super::attitude_mekf::step_diagnostics(
                        mekf,
                        dt,
                        &self.scratch_readings_flat,
                        &self.estimator_config,
                        y,
                        self.nees,
                    )
                    .map_err(str::to_owned)?;
                    self.scratch_estimate.copy_from_slice(&values);
                }
                EstimateState::Ekf(ekf) => {
                    let values = super::attitude_ekf::step_diagnostics(
                        ekf,
                        dt,
                        &self.scratch_readings_flat,
                        &self.estimator_config,
                        y,
                        self.nees,
                    )
                    .map_err(str::to_owned)?;
                    self.scratch_estimate.copy_from_slice(&values);
                }
            }
        }
        Ok(())
    }

    pub fn record(&mut self, sampled_this_step: bool) {
        self.recorded_rows += 1;
        if sampled_this_step {
            for k in 0..8 {
                self.recorded_clean[k].extend_from_slice(&self.scratch_clean[k]);
                self.recorded_reading[k].extend_from_slice(&self.scratch_reading[k]);
            }
            if self.estimate.is_some() {
                self.recorded_estimates
                    .extend_from_slice(&self.scratch_estimate);
            }
        } else {
            for k in 0..8 {
                self.recorded_clean[k]
                    .resize(self.recorded_clean[k].len() + self.counts[k], f64::NAN);
                self.recorded_reading[k]
                    .resize(self.recorded_reading[k].len() + self.counts[k], f64::NAN);
            }
            if self.estimate.is_some() {
                let w = if self.is_ekf { 109 } else { 96 };
                self.recorded_estimates
                    .resize(self.recorded_estimates.len() + w, f64::NAN);
            }
        }
    }
}
