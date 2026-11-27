use adcs_subsystem::controllers::runtime::{
    init_controller_work, step_controller, sync_actuator_enable, ControllerRuntime,
};
use adcs_subsystem::estimator::attitude_estimator::attitude_mekf::SensorConfig as AdcsSensorConfig;
use adcs_subsystem::state_machine::{
    apply_hw_mask, evaluate_goal, init_adcs_work, step_adcs_sm, ADCSSMRuntime,
};
use obc_subsystem::runtime::{init_obc_work, step_obc_core, OBCRuntime};

use super::simulation_sampling::SimulationSampler;
use super::Runtime;

fn apply_obc_state_masks(
    state_id: i64,
    obc_rt: &OBCRuntime,
    sensor_enabled: &mut [bool],
    rw_enabled: &mut [bool],
    mtq_enabled: &mut [bool],
) {
    if state_id < 0 {
        return;
    }
    let idx = state_id as usize;
    if idx >= obc_rt.state_sensor_mask.len() {
        return;
    }
    apply_hw_mask(&obc_rt.state_sensor_mask[idx], sensor_enabled);
    apply_hw_mask(&obc_rt.state_rw_mask[idx], rw_enabled);
    apply_hw_mask(&obc_rt.state_mtq_mask[idx], mtq_enabled);
}

fn apply_obc_tc_masks(
    tc_idx: i64,
    obc_rt: &OBCRuntime,
    sensor_enabled: &mut [bool],
    rw_enabled: &mut [bool],
    mtq_enabled: &mut [bool],
) {
    if tc_idx < 0 {
        return;
    }
    let idx = tc_idx as usize;
    if idx >= obc_rt.tc_trans_sensor_mask.len() {
        return;
    }
    apply_hw_mask(&obc_rt.tc_trans_sensor_mask[idx], sensor_enabled);
    apply_hw_mask(&obc_rt.tc_trans_rw_mask[idx], rw_enabled);
    apply_hw_mask(&obc_rt.tc_trans_mtq_mask[idx], mtq_enabled);
}

fn apply_adcs_mode_masks(
    mode_id: i64,
    adcs_sm_rt: &ADCSSMRuntime,
    sensor_enabled: &mut [bool],
    rw_enabled: &mut [bool],
    mtq_enabled: &mut [bool],
) {
    if !adcs_sm_rt.enabled || mode_id < 0 {
        return;
    }
    let idx = mode_id as usize;
    if idx >= adcs_sm_rt.mode_sensor_mask.len() {
        return;
    }
    apply_hw_mask(&adcs_sm_rt.mode_sensor_mask[idx], sensor_enabled);
    apply_hw_mask(&adcs_sm_rt.mode_rw_mask[idx], rw_enabled);
    apply_hw_mask(&adcs_sm_rt.mode_mtq_mask[idx], mtq_enabled);
}

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
    pub dt_obc: f64,
    pub last_ctrl_time: f64,
    pub last_obc_time: f64,
    pub ctrl_log_flags: u32,
    pub n_wheels: usize,
    pub n_mtqs: usize,

    pub sm_trace: Vec<i64>,
    pub ctrl_log: Vec<f64>,
    pub cmd_rw_log: Vec<f64>,
    pub cmd_mtq_log: Vec<f64>,
    pub recorded_rows: usize,

    wheel_commands: Vec<f64>,
    dipole_commands: Vec<f64>,
    sensor_enabled: Vec<bool>,
    rw_enabled: Vec<bool>,
    mtq_enabled: Vec<bool>,
    base_wheel_max: Vec<f64>,
    base_mtq_max: Vec<f64>,
    scratch_sm: [i64; 3],
    scratch_ctrl: [f64; 20],
    active_obc_mode: i64,
    active_adcs_mode: i64,
    active_adcs_goal: i64,
    pending_cmd_mode: i64,
    pending_cmd_goal: i64,
    prev_ctrl_idx: i64,
    initialized: bool,
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
        dt_obc: f64,
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
        let sensor_enabled = sampler.estimator_config.enabled.clone();
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
            dt_obc,
            last_ctrl_time: 0.0,
            last_obc_time: 0.0,
            ctrl_log_flags,
            n_wheels,
            n_mtqs,
            sm_trace: Vec::new(),
            ctrl_log: Vec::new(),
            cmd_rw_log: Vec::new(),
            cmd_mtq_log: Vec::new(),
            recorded_rows: 0,
            wheel_commands: vec![0.0; n_wheels],
            dipole_commands: vec![0.0; n_mtqs],
            sensor_enabled,
            rw_enabled,
            mtq_enabled,
            base_wheel_max,
            base_mtq_max,
            scratch_sm: [-1, -1, -1],
            scratch_ctrl: [f64::NAN; 20],
            active_obc_mode: -1,
            active_adcs_mode: -1,
            active_adcs_goal: -1,
            pending_cmd_mode: -1,
            pending_cmd_goal: -1,
            prev_ctrl_idx: -1,
            initialized: false,
        }
    }

    fn sync_hardware(
        &mut self,
        sampler: &mut SimulationSampler,
        runtime: &mut Runtime,
    ) {
        for (idx, &en) in self.sensor_enabled.iter().enumerate() {
            sampler.set_sensor_enabled(idx, en);
            if idx < self.sensor_config.enabled.len() {
                self.sensor_config.enabled[idx] = en;
            }
        }
        if self.has_ctrl {
            sync_actuator_enable(
                &self.rw_enabled,
                &self.mtq_enabled,
                &self.base_wheel_max,
                &self.base_mtq_max,
                &mut self.ctrl_rt,
                &mut self.wheel_commands,
                &mut self.dipole_commands,
            );
            let n_rw = runtime.wheel_commands.len().min(self.wheel_commands.len());
            runtime.wheel_commands[..n_rw].copy_from_slice(&self.wheel_commands[..n_rw]);
            let n_mtq = runtime.dipole_commands.len().min(self.dipole_commands.len());
            runtime.dipole_commands[..n_mtq].copy_from_slice(&self.dipole_commands[..n_mtq]);
        }
    }

    pub fn step_state_machines(
        &mut self,
        t: f64,
        y: &[f64],
        sampler: &mut SimulationSampler,
        runtime: &mut Runtime,
        on_obc: bool,
        on_adcs: bool,
    ) {
        let jd = self.epoch_jd + t / 86400.0;
        let mut hw_dirty = false;

        if !self.initialized {
            self.initialized = true;
            if self.has_obc {
                let init_state = self.obc_work[0] as usize;
                self.pending_cmd_mode = self
                    .obc_rt
                    .state_adcs_mode
                    .get(init_state)
                    .copied()
                    .unwrap_or(-1);
                self.pending_cmd_goal = self
                    .obc_rt
                    .state_adcs_goal
                    .get(init_state)
                    .copied()
                    .unwrap_or(-1);
                apply_obc_state_masks(
                    init_state as i64,
                    &self.obc_rt,
                    &mut self.sensor_enabled,
                    &mut self.rw_enabled,
                    &mut self.mtq_enabled,
                );
                let (st, cm, cg, entered_state, fired_tc_idx) = step_obc_core(
                    y,
                    0.0,
                    self.epoch_jd,
                    self.dt_obc,
                    &self.obc_rt,
                    &mut self.obc_work,
                );
                self.active_obc_mode = st as i64;
                if entered_state >= 0 {
                    apply_obc_state_masks(
                        entered_state,
                        &self.obc_rt,
                        &mut self.sensor_enabled,
                        &mut self.rw_enabled,
                        &mut self.mtq_enabled,
                    );
                }
                if fired_tc_idx >= 0 {
                    apply_obc_tc_masks(
                        fired_tc_idx,
                        &self.obc_rt,
                        &mut self.sensor_enabled,
                        &mut self.rw_enabled,
                        &mut self.mtq_enabled,
                    );
                }
                if cm >= 0 {
                    self.pending_cmd_mode = cm;
                }
                if cg >= 0 {
                    self.pending_cmd_goal = cg;
                }
                hw_dirty = true;
            }
            if self.has_adcs_sm {
                let (mode, goal) = step_adcs_sm(
                    0.0,
                    &self.adcs_sm_rt,
                    &mut self.adcs_sm_work,
                    self.pending_cmd_mode,
                    self.pending_cmd_goal,
                );
                self.active_adcs_mode = mode as i64;
                self.active_adcs_goal = goal as i64;
                self.pending_cmd_mode = -1;
                self.pending_cmd_goal = -1;
                apply_adcs_mode_masks(
                    self.active_adcs_mode,
                    &self.adcs_sm_rt,
                    &mut self.sensor_enabled,
                    &mut self.rw_enabled,
                    &mut self.mtq_enabled,
                );
                hw_dirty = true;
            }
        } else {
            if self.has_obc && on_obc {
                let dt_step = if t > self.last_obc_time {
                    t - self.last_obc_time
                } else {
                    self.dt_obc
                };
                self.last_obc_time = t;
                let (st, cm, cg, entered_state, fired_tc_idx) = step_obc_core(
                    y,
                    t,
                    jd,
                    dt_step,
                    &self.obc_rt,
                    &mut self.obc_work,
                );
                self.active_obc_mode = st as i64;
                if entered_state >= 0 {
                    apply_obc_state_masks(
                        entered_state,
                        &self.obc_rt,
                        &mut self.sensor_enabled,
                        &mut self.rw_enabled,
                        &mut self.mtq_enabled,
                    );
                    hw_dirty = true;
                }
                if fired_tc_idx >= 0 {
                    apply_obc_tc_masks(
                        fired_tc_idx,
                        &self.obc_rt,
                        &mut self.sensor_enabled,
                        &mut self.rw_enabled,
                        &mut self.mtq_enabled,
                    );
                    hw_dirty = true;
                }
                if cm >= 0 {
                    self.pending_cmd_mode = cm;
                }
                if cg >= 0 {
                    self.pending_cmd_goal = cg;
                }
            }
            if self.has_adcs_sm && on_adcs {
                let prev_mode = self.active_adcs_mode;
                let (mode, goal) = step_adcs_sm(
                    t,
                    &self.adcs_sm_rt,
                    &mut self.adcs_sm_work,
                    self.pending_cmd_mode,
                    self.pending_cmd_goal,
                );
                self.active_adcs_mode = mode as i64;
                self.active_adcs_goal = goal as i64;
                self.pending_cmd_mode = -1;
                self.pending_cmd_goal = -1;
                if self.active_adcs_mode != prev_mode {
                    apply_adcs_mode_masks(
                        self.active_adcs_mode,
                        &self.adcs_sm_rt,
                        &mut self.sensor_enabled,
                        &mut self.rw_enabled,
                        &mut self.mtq_enabled,
                    );
                    hw_dirty = true;
                }
            }
        }

        if hw_dirty {
            self.sync_hardware(sampler, runtime);
        }

        self.scratch_sm = [
            if self.has_obc { self.active_obc_mode } else { -1 },
            if self.has_adcs_sm { self.active_adcs_mode } else { -1 },
            if self.has_adcs_sm { self.active_adcs_goal } else { -1 },
        ];
    }

    pub fn step_control(
        &mut self,
        t: f64,
        y: &[f64],
        sampler: &SimulationSampler,
        runtime: &mut Runtime,
    ) {
        let jd = self.epoch_jd + t / 86400.0;
        let dt_step = if t > self.last_ctrl_time {
            t - self.last_ctrl_time
        } else {
            self.dt_ctrl
        };
        self.last_ctrl_time = t;

        self.sensor_config
            .reference
            .copy_from_slice(&sampler.estimator_config.reference);

        self.scratch_ctrl.fill(f64::NAN);

        let (q_goal, w_goal) = if self.has_adcs_sm && self.active_adcs_goal > 0 {
            evaluate_goal(
                self.active_adcs_goal as usize,
                &self.adcs_sm_rt,
                y,
                jd,
                self.mu,
            )
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
            let ctrl_idx = if self.has_adcs_sm && self.active_adcs_mode >= 0 {
                self.adcs_sm_rt
                    .mode_default_controller
                    .get(self.active_adcs_mode as usize)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };

            if ctrl_idx as i64 != self.prev_ctrl_idx {
                self.ctrl_work = [0.0; 4];
                self.prev_ctrl_idx = ctrl_idx as i64;
            }

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
        if (self.ctrl_log_flags & 63) != 0 {
            if sampled_this_step {
                self.ctrl_log.extend_from_slice(&self.scratch_ctrl);
            } else {
                self.ctrl_log.extend_from_slice(&[f64::NAN; 20]);
            }
        }
        if (self.ctrl_log_flags & 64) != 0 {
            if sampled_this_step {
                self.cmd_rw_log.extend_from_slice(&self.wheel_commands);
            } else {
                self.cmd_rw_log
                    .resize(self.cmd_rw_log.len() + self.n_wheels, f64::NAN);
            }
        }
        if (self.ctrl_log_flags & 128) != 0 {
            if sampled_this_step {
                self.cmd_mtq_log.extend_from_slice(&self.dipole_commands);
            } else {
                self.cmd_mtq_log
                    .resize(self.cmd_mtq_log.len() + self.n_mtqs, f64::NAN);
            }
        }
    }
}
