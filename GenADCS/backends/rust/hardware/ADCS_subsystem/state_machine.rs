use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::controllers::{Controller, ControllerTable, NoController};
use crate::goals::{Goal, NoGoal, DEFAULT_MU};

pub const GOAL_NONE: i32 = 0;
pub const GOAL_NADIR: i32 = 1;
pub const GOAL_ZENITH: i32 = 2;
pub const GOAL_ALONG_TRACK: i32 = 3;
pub const GOAL_CROSS_TRACK: i32 = 4;
pub const GOAL_VELOCITY: i32 = 5;
pub const GOAL_SUN: i32 = 6;
pub const GOAL_COORDINATE: i32 = 7;
pub const GOAL_COORDINATE_FMC: i32 = 8;
pub const GOAL_QUATERNION: i32 = 9;
pub const GOAL_ANGULAR_RATE: i32 = 10;

pub const GOAL_PARAM_WIDTH: usize = 4;

pub type DeviceMaskSpec = BTreeMap<String, bool>;

#[derive(Clone)]
pub struct GoalTable {
    pub kinds: Vec<i32>,
    pub params: Vec<[f64; GOAL_PARAM_WIDTH]>,
    pub names: Vec<String>,
    pub objects: Vec<Arc<dyn Goal + Send + Sync>>,
}

impl Default for GoalTable {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalTable {
    pub fn new() -> Self {
        Self {
            kinds: vec![GOAL_NONE],
            params: vec![[0.0; GOAL_PARAM_WIDTH]],
            names: vec!["NoGoal".to_string()],
            objects: vec![Arc::new(NoGoal)],
        }
    }

    pub fn register_with_meta(
        &mut self,
        goal: Option<Arc<dyn Goal + Send + Sync>>,
        kind: i32,
        params: [f64; GOAL_PARAM_WIDTH],
        label: String,
    ) -> usize {
        let Some(g) = goal else {
            return 0;
        };
        if kind == GOAL_NONE {
            return 0;
        }
        let idx = self.kinds.len();
        self.kinds.push(kind);
        self.params.push(params);
        self.names.push(label);
        self.objects.push(g);
        idx
    }
}

pub fn compile_sensor_mask(
    spec: &DeviceMaskSpec,
    sensor_names: &[&str],
) -> Result<Vec<i8>, String> {
    let mut mask = vec![-1_i8; sensor_names.len()];
    if spec.is_empty() || sensor_names.is_empty() {
        return Ok(mask);
    }
    for (key, &val) in spec {
        let mut matched = false;
        let code = i8::from(val);
        for (i, &name) in sensor_names.iter().enumerate() {
            if key == "*" || key == "all" || key == "sensors" || key == name {
                mask[i] = code;
                matched = true;
            }
        }
        if !matched {
            return Err(format!("unknown sensor reference in state machine: {key:?}"));
        }
    }
    Ok(mask)
}

pub fn compile_actuator_masks(
    spec: &DeviceMaskSpec,
    rw_names: &[&str],
    mtq_names: &[&str],
) -> Result<(Vec<i8>, Vec<i8>), String> {
    let mut rw_mask = vec![-1_i8; rw_names.len()];
    let mut mtq_mask = vec![-1_i8; mtq_names.len()];
    if spec.is_empty() || (rw_names.is_empty() && mtq_names.is_empty()) {
        return Ok((rw_mask, mtq_mask));
    }
    for (key, &val) in spec {
        let mut matched = false;
        let code = i8::from(val);
        for (i, &name) in rw_names.iter().enumerate() {
            if key == "*"
                || key == "all"
                || key == "actuators"
                || key == "reaction_wheels"
                || key == "ReactionWheel"
                || key == name
            {
                rw_mask[i] = code;
                matched = true;
            }
        }
        for (i, &name) in mtq_names.iter().enumerate() {
            if key == "*"
                || key == "all"
                || key == "actuators"
                || key == "magnetorquers"
                || key == "Magnetorquer"
                || key == name
            {
                mtq_mask[i] = code;
                matched = true;
            }
        }
        if !matched {
            return Err(format!("unknown actuator reference in state machine: {key:?}"));
        }
    }
    Ok((rw_mask, mtq_mask))
}

pub fn apply_hw_mask(mask: &[i8], enabled: &mut [bool]) {
    for (en, &m) in enabled.iter_mut().zip(mask.iter()) {
        if m == 0 {
            *en = false;
        } else if m == 1 {
            *en = true;
        }
    }
}

#[derive(Clone)]
pub struct ADCSMode {
    pub name: String,
    pub goal: Arc<dyn Goal + Send + Sync>,
    pub goal_kind: i32,
    pub goal_params: [f64; GOAL_PARAM_WIDTH],
    pub goal_label: String,
    pub controller: Arc<dyn Controller>,
    pub sensors: DeviceMaskSpec,
    pub actuators: DeviceMaskSpec,
}

impl ADCSMode {
    pub fn new(
        name: String,
        goal: Option<Arc<dyn Goal + Send + Sync>>,
        goal_kind: i32,
        goal_params: [f64; GOAL_PARAM_WIDTH],
        goal_label: String,
        controller: Option<Arc<dyn Controller>>,
        sensors: DeviceMaskSpec,
        actuators: DeviceMaskSpec,
    ) -> Result<Self, String> {
        if name.is_empty() {
            return Err("ADCSMode name must be nonempty".into());
        }
        Ok(Self {
            name,
            goal: goal.unwrap_or_else(|| Arc::new(NoGoal)),
            goal_kind,
            goal_params,
            goal_label,
            controller: controller.unwrap_or_else(|| Arc::new(NoController)),
            sensors,
            actuators,
        })
    }
}

#[derive(Clone)]
pub enum ADCSTimelineTarget {
    ModeName(String),
    GoalSpec {
        goal: Arc<dyn Goal + Send + Sync>,
        kind: i32,
        params: [f64; GOAL_PARAM_WIDTH],
        label: String,
    },
}

#[derive(Clone)]
pub struct ADCSStateMachine {
    pub initial: String,
    pub modes: Vec<ADCSMode>,
    pub timeline: Vec<(f64, ADCSTimelineTarget)>,
}

impl ADCSStateMachine {
    pub fn new(
        initial: String,
        modes: Vec<ADCSMode>,
        timeline: Vec<(f64, ADCSTimelineTarget)>,
    ) -> Result<Self, String> {
        if modes.is_empty() {
            return Err("ADCSStateMachine requires at least one ADCSMode".into());
        }
        let mut names = HashSet::new();
        for m in &modes {
            if !names.insert(m.name.clone()) {
                return Err("duplicate ADCSMode names".into());
            }
        }
        if !names.contains(&initial) {
            return Err(format!("initial ADCS mode {initial:?} not found"));
        }
        for (t, item) in &timeline {
            if !t.is_finite() || *t < 0.0 {
                return Err("ADCS timeline times must be finite and nonnegative".into());
            }
            if let ADCSTimelineTarget::ModeName(ref m_name) = item {
                if !names.contains(m_name) {
                    return Err(format!("unknown ADCS mode in timeline: {m_name:?}"));
                }
            }
        }
        Ok(Self {
            initial,
            modes,
            timeline,
        })
    }
}

#[derive(Clone)]
pub struct ADCSSMRuntime {
    pub enabled: bool,
    pub initial_mode: usize,
    pub mode_default_goal: Vec<usize>,
    pub mode_default_controller: Vec<usize>,
    pub mode_sensor_mask: Vec<Vec<i8>>,
    pub mode_rw_mask: Vec<Vec<i8>>,
    pub mode_mtq_mask: Vec<Vec<i8>>,
    pub tl_times: Vec<f64>,
    pub tl_modes: Vec<i64>,
    pub tl_goals: Vec<i64>,
    pub goal_kinds: Vec<i32>,
    pub goal_params: Vec<[f64; GOAL_PARAM_WIDTH]>,
    pub goal_objects: Vec<Arc<dyn Goal + Send + Sync>>,
    pub mode_names: Vec<String>,
    pub goal_names: Vec<String>,
}

pub fn prepare_adcs_sm(
    adcs_sm: Option<&ADCSStateMachine>,
    goal_table: Option<&mut GoalTable>,
    controller_table: Option<&mut ControllerTable>,
    sensor_names: &[&str],
    rw_names: &[&str],
    mtq_names: &[&str],
) -> Result<ADCSSMRuntime, String> {
    let mut local_gt = GoalTable::new();
    let gt = goal_table.unwrap_or(&mut local_gt);
    let mut local_ct = ControllerTable::new();
    let ct = controller_table.unwrap_or(&mut local_ct);

    let Some(sm) = adcs_sm else {
        return Ok(ADCSSMRuntime {
            enabled: false,
            initial_mode: 0,
            mode_default_goal: vec![0],
            mode_default_controller: vec![0],
            mode_sensor_mask: vec![vec![-1_i8; sensor_names.len()]],
            mode_rw_mask: vec![vec![-1_i8; rw_names.len()]],
            mode_mtq_mask: vec![vec![-1_i8; mtq_names.len()]],
            tl_times: vec![f64::INFINITY],
            tl_modes: vec![-1],
            tl_goals: vec![-1],
            goal_kinds: gt.kinds.clone(),
            goal_params: gt.params.clone(),
            goal_objects: gt.objects.clone(),
            mode_names: vec!["default".to_string()],
            goal_names: gt.names.clone(),
        });
    };

    let mut mode_to_id = BTreeMap::new();
    let n_modes = sm.modes.len();
    let mut mode_names = Vec::with_capacity(n_modes);
    let mut mode_default_goal = Vec::with_capacity(n_modes);
    let mut mode_default_controller = Vec::with_capacity(n_modes);
    let mut mode_sensor_mask = vec![vec![-1_i8; sensor_names.len()]; n_modes];
    let mut mode_rw_mask = vec![vec![-1_i8; rw_names.len()]; n_modes];
    let mut mode_mtq_mask = vec![vec![-1_i8; mtq_names.len()]; n_modes];

    for (idx, m) in sm.modes.iter().enumerate() {
        mode_to_id.insert(m.name.clone(), idx);
        mode_names.push(m.name.clone());
        let g_idx = gt.register_with_meta(
            Some(m.goal.clone()),
            m.goal_kind,
            m.goal_params,
            m.goal_label.clone(),
        );
        mode_default_goal.push(g_idx);
        mode_default_controller.push(ct.register(Some(m.controller.clone())));
        if !m.sensors.is_empty() {
            mode_sensor_mask[idx] = compile_sensor_mask(&m.sensors, sensor_names)?;
        }
        if !m.actuators.is_empty() {
            let (rw_m, mtq_m) = compile_actuator_masks(&m.actuators, rw_names, mtq_names)?;
            mode_rw_mask[idx] = rw_m;
            mode_mtq_mask[idx] = mtq_m;
        }
    }

    let mut sorted_tl = sm.timeline.clone();
    sorted_tl.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut tl_times = Vec::with_capacity(sorted_tl.len() + 1);
    let mut tl_modes = Vec::with_capacity(sorted_tl.len() + 1);
    let mut tl_goals = Vec::with_capacity(sorted_tl.len() + 1);
    for (t, item) in sorted_tl {
        tl_times.push(t);
        match item {
            ADCSTimelineTarget::ModeName(name) => {
                let mid = mode_to_id[&name];
                tl_modes.push(mid as i64);
                tl_goals.push(mode_default_goal[mid] as i64);
            }
            ADCSTimelineTarget::GoalSpec {
                goal,
                kind,
                params,
                label,
            } => {
                tl_modes.push(-1);
                tl_goals.push(gt.register_with_meta(Some(goal), kind, params, label) as i64);
            }
        }
    }
    tl_times.push(f64::INFINITY);
    tl_modes.push(-1);
    tl_goals.push(-1);

    Ok(ADCSSMRuntime {
        enabled: true,
        initial_mode: mode_to_id[&sm.initial],
        mode_default_goal,
        mode_default_controller,
        mode_sensor_mask,
        mode_rw_mask,
        mode_mtq_mask,
        tl_times,
        tl_modes,
        tl_goals,
        goal_kinds: gt.kinds.clone(),
        goal_params: gt.params.clone(),
        goal_objects: gt.objects.clone(),
        mode_names,
        goal_names: gt.names.clone(),
    })
}

pub fn init_adcs_work(runtime: &ADCSSMRuntime) -> [usize; 3] {
    [
        runtime.initial_mode,
        runtime.mode_default_goal[runtime.initial_mode],
        0,
    ]
}

pub fn step_adcs_sm(
    elapsed_s: f64,
    adcs_sm: &ADCSSMRuntime,
    work: &mut [usize; 3],
    cmd_mode: i64,
    cmd_goal: i64,
) -> (usize, usize) {
    let mut mode = work[0];
    let mut goal = work[1];

    if cmd_mode >= 0 && (cmd_mode as usize) != mode {
        mode = cmd_mode as usize;
        goal = adcs_sm.mode_default_goal[mode];
    }
    if cmd_goal >= 0 {
        goal = cmd_goal as usize;
    }

    let mut tl_idx = work[2];
    while elapsed_s >= adcs_sm.tl_times[tl_idx] {
        let next_m = adcs_sm.tl_modes[tl_idx];
        let next_g = adcs_sm.tl_goals[tl_idx];
        if next_m >= 0 {
            mode = next_m as usize;
        }
        if next_g >= 0 {
            goal = next_g as usize;
        }
        tl_idx += 1;
    }
    work[2] = tl_idx;
    work[0] = mode;
    work[1] = goal;
    (mode, goal)
}

pub fn evaluate_goal(
    goal_idx: usize,
    adcs_sm: &ADCSSMRuntime,
    state: &[f64],
    epoch_jd: f64,
    mu: f64,
) -> ([f64; 4], [f64; 3]) {
    let nan_q = [f64::NAN; 4];
    let nan_w = [f64::NAN; 3];
    if goal_idx == 0 || goal_idx >= adcs_sm.goal_objects.len() {
        return (nan_q, nan_w);
    }
    match adcs_sm.goal_objects[goal_idx].reference(state, epoch_jd, mu) {
        Ok((q_opt, w_opt)) => (q_opt.unwrap_or(nan_q), w_opt.unwrap_or(nan_w)),
        Err(_) => (nan_q, nan_w),
    }
}

pub fn evaluate_goal_default_mu(
    goal_idx: usize,
    adcs_sm: &ADCSSMRuntime,
    state: &[f64],
    epoch_jd: f64,
) -> ([f64; 4], [f64; 3]) {
    evaluate_goal(goal_idx, adcs_sm, state, epoch_jd, DEFAULT_MU)
}
