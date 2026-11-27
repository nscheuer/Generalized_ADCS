use adcs_subsystem::controllers::runtime::{
    init_controller_work, step_controller, sync_actuator_enable, ControllerRuntime,
};
use adcs_subsystem::estimator::attitude_estimator::attitude_mekf::SensorConfig as AdcsSensorConfig;
use adcs_subsystem::state_machine::{
    apply_hw_mask, evaluate_goal, init_adcs_work, step_adcs_sm, ADCSSMRuntime,
};
use obc_subsystem::runtime::{init_obc_work, step_obc, OBCRuntime};

use super::simulation_sampling::SimulationSampler;
use super::Runtime;

pub struct ControlLoopContext {
    pub has_obc: bool,
    pub obc_rt: OBCRuntime,
    pub obc_work: [f64; 5],

    pub has_adcs_sm: bool,
    pub adcs_sm_rt: ADCSSMRuntime,
    pub adcs_sm_work: [usize; 3],

    pub has_ctrl: bool,
    pub ctrl_rt: ControllerRuntime,
    pub ctrl_work: [f64; 4],

    pub sensor_config: AdcsSensorConfig,
    pub mu: f64,
    pub epoch_jd: f64,
    pub dt_ctrl: f64,
    pub last_ctrl_time: f64,
    pub ctrl_log_flags: u32,

    pub sm_trace: Vec<i64>,
    pub ctrl_log: Vec<f64>,
    pub recorded_rows: usize,

    wheel_commands: Vec<f64>,
    dipole_commands: Vec<f64>,
    rw_enabled: Vec<bool>,
    mtq_enabled: Vec<bool>,
    base_wheel_max: Vec<f64>,
    base_mtq_max: Vec<f64>,
    scratch_sm: [i64; 3],
    scratch_ctrl: [f64; 20],
    last_applied_mode: i64,
}

impl ControlLoopContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        has_obc: bool,
        obc_rt: OBCRuntime,
        has_adcs_sm: bool,
        adcs_sm_rt: ADCSSMRuntime,
        has_ctrl: bool,
        ctrl_rt: ControllerRuntime,
        sampler: &SimulationSampler,
        mu: f64,
        epoch_jd: f64,
        dt_ctrl: f64,
        ctrl_log_flags: u32,
        n_wheels: usize,
        n_mtqs: usize,
    ) -> Self {
        let obc_work = if has_obc {
            init_obc_work(&obc_rt)
        } else {
            [0.0; 5]
        };
        let adcs_sm_work = if has_adcs_sm {
            init_adcs_work(&adcs_sm_rt)
        } else {
            [0; 3]
        };
        let ctrl_work = init_controller_work();
        let base_wheel_max = ctrl_rt.wheel_max_torque.clone();
        let base_mtq_max = ctrl_rt.mtq_max_dipole.clone();
        let rw_enabled = vec![true; n_wheels];
        let mtq_enabled = vec![true; n_mtqs];
        let sensor_config = AdcsSensorConfig {
            code: sampler.estimator_config.code.clone(),
            orientation: sampler.estimator_config.orientation.clone(),
            misalignment: sampler.estimator_config.misalignment.clone(),
            bias: sampler.estimator_config.bias.clone(),
            noise_cholesky: sampler.estimator_config.noise_cholesky.clone(),
            normal: sampler.estimator_config.normal.clone(),
            fov: sampler.estimator_config.fov.clone(),
            scalars: sampler.estimator_config.scalars.clone(),
            reference: sampler.estimator_config.reference.clone(),
            enabled: sampler.estimator_config.enabled.clone(),
            offsets: sampler.estimator_config.offsets.clone(),
        };
        Self {
            has_obc,
            obc_rt,
            obc_work,
            has_adcs_sm,
            adcs_sm_rt,
            adcs_sm_work,
            has_ctrl,
            ctrl_rt,
            ctrl_work,
            sensor_config,
            mu,
            epoch_jd,
            dt_ctrl,
            last_ctrl_time: 0.0,
            ctrl_log_flags,
            sm_trace: Vec::new(),
            ctrl_log: Vec::new(),
            recorded_rows: 0,
            wheel_commands: vec![0.0; n_wheels],
            dipole_commands: vec![0.0; n_mtqs],
            rw_enabled,
            mtq_enabled,
            base_wheel_max,
            base_mtq_max,
            scratch_sm: [-1, -1, -1],
            scratch_ctrl: [f64::NAN; 20],
            last_applied_mode: -1,
        }
    }

    pub fn active(&self) -> bool {
        self.has_obc || self.has_adcs_sm || self.has_ctrl
    }

    pub fn step(
        &mut self,
        t: f64,
        y: &[f64],
        sampler: &mut SimulationSampler,
        runtime: &mut Runtime,
    ) {
        let jd = self.epoch_jd + t / 86400.0;
        let dt_step = if t > self.last_ctrl_time {
            t - self.last_ctrl_time
        } else {
            self.dt_ctrl
        };
        self.last_ctrl_time = t;

        // Sync reference vectors from sampler into our sensor_config.
        self.sensor_config
            .reference
            .copy_from_slice(&sampler.estimator_config.reference);

        let mut obc_mode: i64 = -1;
        let mut cmd_mode: i64 = -1;
        let mut cmd_goal: i64 = -1;
        if self.has_obc {
            let (st, m, g) = step_obc(y, t, jd, dt_step, &self.obc_rt, &mut self.obc_work);
            obc_mode = st as i64;
            cmd_mode = m;
            cmd_goal = g;
        }

        let mut adcs_mode: i64 = -1;
        let mut adcs_goal: i64 = -1;
        if self.has_adcs_sm {
            let (mode, goal) = step_adcs_sm(
                t,
                &self.adcs_sm_rt,
                &mut self.adcs_sm_work,
                cmd_mode,
                cmd_goal,
            );
            adcs_mode = mode as i64;
            adcs_goal = goal as i64;
            if adcs_mode != self.last_applied_mode {
                if let Some(mask) = self.adcs_sm_rt.mode_sensor_mask.get(mode) {
                    for (idx, &flag) in mask.iter().enumerate() {
                        if flag == 0 {
                            sampler.set_sensor_enabled(idx, false);
                            if idx < self.sensor_config.enabled.len() {
                                self.sensor_config.enabled[idx] = false;
                            }
                        } else if flag == 1 {
                            sampler.set_sensor_enabled(idx, true);
                            if idx < self.sensor_config.enabled.len() {
                                self.sensor_config.enabled[idx] = true;
                            }
                        }
                    }
                }
                if self.has_ctrl {
                    if let Some(rw_mask) = self.adcs_sm_rt.mode_rw_mask.get(mode) {
                        apply_hw_mask(rw_mask, &mut self.rw_enabled);
                    }
                    if let Some(mtq_mask) = self.adcs_sm_rt.mode_mtq_mask.get(mode) {
                        apply_hw_mask(mtq_mask, &mut self.mtq_enabled);
                    }
                    sync_actuator_enable(
                        &self.rw_enabled,
                        &self.mtq_enabled,
                        &self.base_wheel_max,
                        &self.base_mtq_max,
                        &mut self.ctrl_rt,
                        &mut self.wheel_commands,
                        &mut self.dipole_commands,
                    );
                }
                self.last_applied_mode = adcs_mode;
            }
        }

        self.scratch_sm = [obc_mode, adcs_mode, adcs_goal];
        self.scratch_ctrl.fill(f64::NAN);

        let (q_goal, w_goal) = if self.has_adcs_sm && adcs_goal > 0 {
            evaluate_goal(adcs_goal as usize, &self.adcs_sm_rt, y, jd, self.mu)
        } else {
            ([f64::NAN; 4], [f64::NAN; 3])
        };
        if (self.ctrl_log_flags & 1) != 0 {
            self.scratch_ctrl[0..4].copy_from_slice(&q_goal);
        }
        if (self.ctrl_log_flags & 2) != 0 {
            self.scratch_ctrl[4..7].copy_from_slice(&w_goal);
        }

        if self.has_ctrl {
            let ctrl_idx = if self.has_adcs_sm && adcs_mode >= 0 {
                self.adcs_sm_rt
                    .mode_default_controller
                    .get(adcs_mode as usize)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };

            let mut q_est = [y[0], y[1], y[2], y[3]];
            let mut w_est = [y[4], y[5], y[6]];
            if sampler.has_estimator() {
                let est = sampler.estimate_slice();
                if est.len() >= 7
                    && est[0].is_finite()
                    && est[1].is_finite()
                    && est[2].is_finite()
                    && est[3].is_finite()
                {
                    q_est = [est[0], est[1], est[2], est[3]];
                }
                if est.len() >= 7
                    && est[4].is_finite()
                    && est[5].is_finite()
                    && est[6].is_finite()
                {
                    w_est = [est[4], est[5], est[6]];
                }
            }

            let wheel_momenta = if y.len() > 13 { &y[13..] } else { &[] };
            let (tau_ff, q_ff, w_ff, tau_ref) = step_controller(
                ctrl_idx,
                dt_step,
                q_est,
                w_est,
                q_goal,
                w_goal,
                wheel_momenta,
                sampler.readings_flat(),
                &self.sensor_config,
                &self.ctrl_rt,
                &mut self.ctrl_work,
                &mut self.wheel_commands,
                &mut self.dipole_commands,
            );

            if (self.ctrl_log_flags & 4) != 0 {
                self.scratch_ctrl[7..10].copy_from_slice(&tau_ff);
            }
            if (self.ctrl_log_flags & 8) != 0 {
                self.scratch_ctrl[10..14].copy_from_slice(&q_ff);
            }
            if (self.ctrl_log_flags & 16) != 0 {
                self.scratch_ctrl[14..17].copy_from_slice(&w_ff);
            }
            if (self.ctrl_log_flags & 32) != 0 {
                self.scratch_ctrl[17..20].copy_from_slice(&tau_ref);
            }

            let n_rw = runtime.wheel_commands.len().min(self.wheel_commands.len());
            runtime.wheel_commands[..n_rw].copy_from_slice(&self.wheel_commands[..n_rw]);
            let n_mtq = runtime.dipole_commands.len().min(self.dipole_commands.len());
            runtime.dipole_commands[..n_mtq].copy_from_slice(&self.dipole_commands[..n_mtq]);
        }
    }

    pub fn record(&mut self, sampled_this_step: bool) {
        self.recorded_rows += 1;
        if self.has_obc || self.has_adcs_sm {
            if sampled_this_step {
                self.sm_trace.extend_from_slice(&self.scratch_sm);
            } else {
                self.sm_trace.extend_from_slice(&[-1, -1, -1]);
            }
        }
        if self.ctrl_log_flags != 0 {
            if sampled_this_step {
                self.ctrl_log.extend_from_slice(&self.scratch_ctrl);
            } else {
                self.ctrl_log.extend_from_slice(&[f64::NAN; 20]);
            }
        }
    }
}
