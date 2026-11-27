use std::collections::BTreeMap;
use std::sync::Arc;

use adcs_subsystem::goals::{CoordinateGoal, Goal, NoGoal};
use frames_native::ecef_to_gcrf_matrix_native;
use nalgebra::Vector3;

use super::obc_subsystem::OBCSubsystem;
use super::state_machine::{ADCSBinding, HardwareMaskSpec, Mode, Trigger};

pub const FDIR_ANGULAR_RATE: i64 = 1;
pub const OMEGA_EARTH: f64 = 7.2921159e-5;

#[derive(Clone, Default)]
pub struct GoalTable {
    pub objects: Vec<Arc<dyn Goal + Send + Sync>>,
    pub names: Vec<String>,
}

impl GoalTable {
    pub fn new() -> Self {
        Self {
            objects: vec![Arc::new(NoGoal)],
            names: vec!["NoGoal".to_string()],
        }
    }

    pub fn register_named(
        &mut self,
        goal: Option<Arc<dyn Goal + Send + Sync>>,
        type_name: &str,
    ) -> usize {
        let Some(g) = goal else {
            return 0;
        };
        if type_name == "NoGoal" {
            return 0;
        }
        let idx = self.objects.len();
        self.objects.push(g);
        self.names.push(type_name.to_string());
        idx
    }
}

#[derive(Clone, Debug)]
pub struct OBCRuntime {
    pub enabled: bool,
    pub initial_state: usize,
    pub n_states: usize,
    pub tc_trans_start: Vec<usize>,
    pub tc_trans_end: Vec<usize>,
    pub tc_trans_code: Vec<i64>,
    pub tc_trans_target: Vec<usize>,
    pub tc_trans_goal: Vec<i64>,
    pub fov_trans_start: Vec<usize>,
    pub fov_trans_end: Vec<usize>,
    pub fov_ecef: Vec<[f64; 3]>,
    pub fov_cos_limit: Vec<f64>,
    pub fov_negate: Vec<bool>,
    pub fov_target: Vec<usize>,
    pub fov_goal: Vec<i64>,
    pub state_duration: Vec<f64>,
    pub state_auto_target: Vec<i64>,
    pub state_adcs_mode: Vec<i64>,
    pub state_adcs_goal: Vec<i64>,
    pub state_sensor_mask: Vec<Vec<i8>>,
    pub state_rw_mask: Vec<Vec<i8>>,
    pub state_mtq_mask: Vec<Vec<i8>>,
    pub tc_trans_sensor_mask: Vec<Vec<i8>>,
    pub tc_trans_rw_mask: Vec<Vec<i8>>,
    pub tc_trans_mtq_mask: Vec<Vec<i8>>,
    pub tc_times: Vec<f64>,
    pub tc_codes: Vec<i64>,
    pub tc_goals: Vec<i64>,
    pub fdir_enabled: bool,
    pub fdir_safe_state: usize,
    pub fdir_monitor_kinds: Vec<i64>,
    pub fdir_monitor_thresholds: Vec<f64>,
    pub fdir_monitor_targets: Vec<usize>,
    pub fdir_trans_start: Vec<usize>,
    pub fdir_trans_end: Vec<usize>,
    pub fdir_trans_monitor: Vec<i64>,
    pub fdir_trans_target: Vec<usize>,
    pub mode_names: Vec<String>,
}

fn disabled_runtime() -> OBCRuntime {
    OBCRuntime {
        enabled: false,
        initial_state: 0,
        n_states: 0,
        tc_trans_start: Vec::new(),
        tc_trans_end: Vec::new(),
        tc_trans_code: Vec::new(),
        tc_trans_target: Vec::new(),
        tc_trans_goal: Vec::new(),
        fov_trans_start: Vec::new(),
        fov_trans_end: Vec::new(),
        fov_ecef: Vec::new(),
        fov_cos_limit: Vec::new(),
        fov_negate: Vec::new(),
        fov_target: Vec::new(),
        fov_goal: Vec::new(),
        state_duration: Vec::new(),
        state_auto_target: Vec::new(),
        state_adcs_mode: Vec::new(),
        state_adcs_goal: Vec::new(),
        state_sensor_mask: Vec::new(),
        state_rw_mask: Vec::new(),
        state_mtq_mask: Vec::new(),
        tc_trans_sensor_mask: Vec::new(),
        tc_trans_rw_mask: Vec::new(),
        tc_trans_mtq_mask: Vec::new(),
        tc_times: vec![f64::INFINITY],
        tc_codes: vec![-1],
        tc_goals: vec![-1],
        fdir_enabled: false,
        fdir_safe_state: 0,
        fdir_monitor_kinds: Vec::new(),
        fdir_monitor_thresholds: Vec::new(),
        fdir_monitor_targets: Vec::new(),
        fdir_trans_start: Vec::new(),
        fdir_trans_end: Vec::new(),
        fdir_trans_monitor: Vec::new(),
        fdir_trans_target: Vec::new(),
        mode_names: Vec::new(),
    }
}

fn resolve_target(
    name: &str,
    modes_by_name: &BTreeMap<String, &Mode>,
    leaf_ids: &BTreeMap<String, usize>,
) -> Result<usize, String> {
    if let Some(&id) = leaf_ids.get(name) {
        return Ok(id);
    }
    if let Some(mode) = modes_by_name.get(name) {
        if !mode.submodes.is_empty() {
            if let Some(ref sub) = mode.initial_submode {
                let full = format!("{name}.{sub}");
                if let Some(&id) = leaf_ids.get(&full) {
                    return Ok(id);
                }
            }
        }
    }
    Err(format!("unknown target mode {name:?}"))
}

fn resolve_sources(
    pattern: &str,
    modes_by_name: &BTreeMap<String, &Mode>,
    leaf_names: &[String],
    leaf_ids: &BTreeMap<String, usize>,
) -> Result<Vec<usize>, String> {
    if pattern == "*" {
        return Ok((0..leaf_names.len()).collect());
    }
    if let Some(prefix_base) = pattern.strip_suffix(".*") {
        let prefix = format!("{prefix_base}.");
        let matched: Vec<usize> = leaf_names
            .iter()
            .filter(|n| n.starts_with(&prefix))
            .map(|n| leaf_ids[n])
            .collect();
        if matched.is_empty() {
            return Err(format!("transition source {pattern:?} matched no modes"));
        }
        return Ok(matched);
    }
    if let Some(&id) = leaf_ids.get(pattern) {
        return Ok(vec![id]);
    }
    if let Some(mode) = modes_by_name.get(pattern) {
        if !mode.submodes.is_empty() {
            let prefix = format!("{pattern}.");
            return Ok(leaf_names
                .iter()
                .filter(|n| n.starts_with(&prefix))
                .map(|n| leaf_ids[n])
                .collect());
        }
    }
    Err(format!("unknown transition source {pattern:?}"))
}

fn compile_mask(
    spec: Option<&HardwareMaskSpec>,
    names: &[&str],
    aliases: &[&str],
    other_names: &[&str],
    other_aliases: &[&str],
) -> Result<Vec<i8>, String> {
    let mut mask = vec![-1_i8; names.len()];
    let Some(map) = spec else {
        return Ok(mask);
    };
    for (key, &val) in map {
        if aliases.contains(&key.as_str()) {
            mask.fill(i8::from(val));
            continue;
        }
        let mut matched = false;
        for (i, &dev) in names.iter().enumerate() {
            if dev == key {
                mask[i] = i8::from(val);
                matched = true;
            }
        }
        if !matched
            && !other_names.contains(&key.as_str())
            && !other_aliases.contains(&key.as_str())
        {
            return Err(format!("unknown hardware reference in state machine: {key:?}"));
        }
    }
    Ok(mask)
}

pub fn prepare_obc(
    obc: Option<&OBCSubsystem>,
    adcs_mode_to_id: &BTreeMap<String, usize>,
    goal_table: Option<&mut GoalTable>,
    sensor_names: &[&str],
    rw_names: &[&str],
    mtq_names: &[&str],
) -> Result<OBCRuntime, String> {
    let Some(obc) = obc else {
        return Ok(disabled_runtime());
    };
    let Some(ref sm) = obc.state_machine else {
        return Ok(disabled_runtime());
    };

    let mut local_gt = GoalTable::new();
    let gt = goal_table.unwrap_or(&mut local_gt);

    let mut modes_by_name = BTreeMap::new();
    for m in &sm.modes {
        modes_by_name.insert(m.name.clone(), m);
    }

    let mut leaf_names = Vec::new();
    let mut leaf_modes: Vec<&Mode> = Vec::new();
    let mut leaf_parents: Vec<Option<&Mode>> = Vec::new();
    for m in &sm.modes {
        if !m.submodes.is_empty() {
            for sub in &m.submodes {
                leaf_names.push(format!("{}.{}", m.name, sub.name));
                leaf_modes.push(sub);
                leaf_parents.push(Some(m));
            }
        } else {
            leaf_names.push(m.name.clone());
            leaf_modes.push(m);
            leaf_parents.push(None);
        }
    }

    let mut leaf_ids = BTreeMap::new();
    for (idx, name) in leaf_names.iter().enumerate() {
        leaf_ids.insert(name.clone(), idx);
    }
    let n_states = leaf_names.len();
    let initial_state = resolve_target(&sm.initial, &modes_by_name, &leaf_ids)?;

    let mut cmd_to_code: BTreeMap<String, i64> = BTreeMap::new();
    let mut get_cmd_code = |cmd: &str| -> i64 {
        if let Some(&c) = cmd_to_code.get(cmd) {
            c
        } else {
            let c = cmd_to_code.len() as i64;
            cmd_to_code.insert(cmd.to_string(), c);
            c
        }
    };

    let mut state_duration = vec![-1.0; n_states];
    let mut state_auto_target = vec![-1_i64; n_states];
    let mut state_adcs_mode = vec![-1_i64; n_states];
    let mut state_adcs_goal = vec![-1_i64; n_states];
    let mut state_sensor_mask = vec![vec![-1_i8; sensor_names.len()]; n_states];
    let mut state_rw_mask = vec![vec![-1_i8; rw_names.len()]; n_states];
    let mut state_mtq_mask = vec![vec![-1_i8; mtq_names.len()]; n_states];

    for idx in 0..n_states {
        let leaf = leaf_modes[idx];
        let parent = leaf_parents[idx];

        if let Some(dur) = leaf.duration {
            state_duration[idx] = dur;
            if let Some(ref auto_t) = leaf.auto_target {
                let mut target_name = auto_t.clone();
                if let Some(p) = parent {
                    if !target_name.contains('.') {
                        let scoped = format!("{}.{}", p.name, target_name);
                        if leaf_ids.contains_key(&scoped) {
                            target_name = scoped;
                        }
                    }
                }
                state_auto_target[idx] =
                    resolve_target(&target_name, &modes_by_name, &leaf_ids)? as i64;
            }
        }

        let binding = leaf
            .bindings
            .adcs
            .as_ref()
            .or_else(|| parent.and_then(|p| p.bindings.adcs.as_ref()));
        if let Some(b) = binding {
            match b {
                ADCSBinding::ModeName(m_name) => {
                    let Some(&id) = adcs_mode_to_id.get(m_name) else {
                        return Err(format!(
                            "OBC binding references unknown ADCS mode {m_name:?}"
                        ));
                    };
                    state_adcs_mode[idx] = id as i64;
                }
                ADCSBinding::GoalOnly(g) => {
                    state_adcs_goal[idx] = gt.register_named(Some(g.clone()), "Goal") as i64;
                }
                ADCSBinding::ModeAndGoal(m_name, g) => {
                    let Some(&id) = adcs_mode_to_id.get(m_name) else {
                        return Err(format!(
                            "OBC binding references unknown ADCS mode {m_name:?}"
                        ));
                    };
                    state_adcs_mode[idx] = id as i64;
                    state_adcs_goal[idx] = gt.register_named(Some(g.clone()), "Goal") as i64;
                }
            }
        }

        let sensor_binding = leaf
            .bindings
            .sensors
            .as_ref()
            .or_else(|| parent.and_then(|p| p.bindings.sensors.as_ref()));
        if sensor_binding.is_some() {
            state_sensor_mask[idx] = compile_mask(
                sensor_binding,
                sensor_names,
                &["*", "all", "sensors"],
                &[],
                &[],
            )?;
        }

        let actuator_binding = leaf
            .bindings
            .actuators
            .as_ref()
            .or_else(|| parent.and_then(|p| p.bindings.actuators.as_ref()));
        if actuator_binding.is_some() {
            state_rw_mask[idx] = compile_mask(
                actuator_binding,
                rw_names,
                &["*", "all", "actuators", "reaction_wheels", "ReactionWheel"],
                mtq_names,
                &["magnetorquers", "Magnetorquer"],
            )?;
            state_mtq_mask[idx] = compile_mask(
                actuator_binding,
                mtq_names,
                &["*", "all", "actuators", "magnetorquers", "Magnetorquer"],
                rw_names,
                &["reaction_wheels", "ReactionWheel"],
            )?;
        }
    }

    let mut monitor_to_id = BTreeMap::new();
    if let Some(ref fdir) = obc.fdir {
        for (idx, mon) in fdir.monitors.iter().enumerate() {
            monitor_to_id.insert(mon.name().to_string(), idx);
        }
    }

    struct TcRec {
        priority: i32,
        code: i64,
        target: usize,
        goal: i64,
        s_mask: Vec<i8>,
        rw_mask: Vec<i8>,
        mtq_mask: Vec<i8>,
    }
    struct FovRec {
        priority: i32,
        ecef: [f64; 3],
        cos_limit: f64,
        negate: bool,
        target: usize,
        goal: i64,
    }
    struct FdirRec {
        priority: i32,
        mon_idx: i64,
        target: usize,
    }

    let mut per_state_tc: Vec<Vec<TcRec>> = (0..n_states).map(|_| Vec::new()).collect();
    let mut per_state_fov: Vec<Vec<FovRec>> = (0..n_states).map(|_| Vec::new()).collect();
    let mut per_state_fdir: Vec<Vec<FdirRec>> = (0..n_states).map(|_| Vec::new()).collect();

    for tr in &sm.transitions {
        let sources = resolve_sources(&tr.source, &modes_by_name, &leaf_names, &leaf_ids)?;
        let target_id = resolve_target(&tr.target, &modes_by_name, &leaf_ids)?;
        match &tr.trigger {
            Trigger::Telecommand(tc) => {
                let code = get_cmd_code(&tc.command);
                let g_idx = if let Some(ref g) = tc.goal {
                    gt.register_named(Some(g.clone()), "Goal") as i64
                } else {
                    -1
                };
                let s_m = compile_mask(
                    tc.sensors.as_ref(),
                    sensor_names,
                    &["*", "all", "sensors"],
                    &[],
                    &[],
                )?;
                let rw_m = compile_mask(
                    tc.actuators.as_ref(),
                    rw_names,
                    &["*", "all", "actuators", "reaction_wheels", "ReactionWheel"],
                    mtq_names,
                    &["magnetorquers", "Magnetorquer"],
                )?;
                let mtq_m = compile_mask(
                    tc.actuators.as_ref(),
                    mtq_names,
                    &["*", "all", "actuators", "magnetorquers", "Magnetorquer"],
                    rw_names,
                    &["reaction_wheels", "ReactionWheel"],
                )?;
                for src in sources {
                    per_state_tc[src].push(TcRec {
                        priority: tr.priority,
                        code,
                        target: target_id,
                        goal: g_idx,
                        s_mask: s_m.clone(),
                        rw_mask: rw_m.clone(),
                        mtq_mask: mtq_m.clone(),
                    });
                }
            }
            Trigger::InFOV(fov) => {
                let coord = CoordinateGoal::new(fov.lat, fov.lon, Some(fov.alt))?;
                let cos_limit = fov.max_off_nadir_deg.to_radians().cos();
                let g_idx = if let Some(ref g) = fov.goal {
                    gt.register_named(Some(g.clone()), "Goal") as i64
                } else if !fov.negate {
                    gt.register_named(Some(Arc::new(coord.clone())), "CoordinateGoal") as i64
                } else {
                    -1
                };
                for src in sources {
                    per_state_fov[src].push(FovRec {
                        priority: tr.priority,
                        ecef: coord.target_ecef_m,
                        cos_limit,
                        negate: fov.negate,
                        target: target_id,
                        goal: g_idx,
                    });
                }
            }
            Trigger::Completion(_) => {
                for src in sources {
                    if state_duration[src] <= 0.0 {
                        return Err(format!(
                            "Completion transition from {:?} requires Mode.duration",
                            leaf_names[src]
                        ));
                    }
                    state_auto_target[src] = target_id as i64;
                }
            }
            Trigger::After(after) => {
                for src in sources {
                    state_duration[src] = after.duration;
                    state_auto_target[src] = target_id as i64;
                }
            }
            Trigger::FDIRTriggered(fdir_tr) => {
                let mon_idx = match &fdir_tr.monitor {
                    None => -1,
                    Some(name) => match monitor_to_id.get(name) {
                        Some(&id) => id as i64,
                        None => {
                            return Err(format!(
                                "FDIRTriggered references unknown monitor {name:?}"
                            ))
                        }
                    },
                };
                for src in sources {
                    per_state_fdir[src].push(FdirRec {
                        priority: tr.priority,
                        mon_idx,
                        target: target_id,
                    });
                }
            }
        }
    }

    let mut tc_trans_start = vec![0; n_states];
    let mut tc_trans_end = vec![0; n_states];
    let mut tc_trans_code = Vec::new();
    let mut tc_trans_target = Vec::new();
    let mut tc_trans_goal = Vec::new();
    let mut tc_trans_sensor_mask = Vec::new();
    let mut tc_trans_rw_mask = Vec::new();
    let mut tc_trans_mtq_mask = Vec::new();
    let mut cursor = 0;
    for s in 0..n_states {
        tc_trans_start[s] = cursor;
        per_state_tc[s].sort_by(|a, b| b.priority.cmp(&a.priority));
        for e in &per_state_tc[s] {
            tc_trans_code.push(e.code);
            tc_trans_target.push(e.target);
            tc_trans_goal.push(e.goal);
            tc_trans_sensor_mask.push(e.s_mask.clone());
            tc_trans_rw_mask.push(e.rw_mask.clone());
            tc_trans_mtq_mask.push(e.mtq_mask.clone());
            cursor += 1;
        }
        tc_trans_end[s] = cursor;
    }

    let mut fov_trans_start = vec![0; n_states];
    let mut fov_trans_end = vec![0; n_states];
    let mut fov_ecef = Vec::new();
    let mut fov_cos_limit = Vec::new();
    let mut fov_negate = Vec::new();
    let mut fov_target = Vec::new();
    let mut fov_goal = Vec::new();
    cursor = 0;
    for s in 0..n_states {
        fov_trans_start[s] = cursor;
        per_state_fov[s].sort_by(|a, b| b.priority.cmp(&a.priority));
        for e in &per_state_fov[s] {
            fov_ecef.push(e.ecef);
            fov_cos_limit.push(e.cos_limit);
            fov_negate.push(e.negate);
            fov_target.push(e.target);
            fov_goal.push(e.goal);
            cursor += 1;
        }
        fov_trans_end[s] = cursor;
    }

    let mut fdir_trans_start = vec![0; n_states];
    let mut fdir_trans_end = vec![0; n_states];
    let mut fdir_trans_monitor = Vec::new();
    let mut fdir_trans_target = Vec::new();
    cursor = 0;
    for s in 0..n_states {
        fdir_trans_start[s] = cursor;
        per_state_fdir[s].sort_by(|a, b| b.priority.cmp(&a.priority));
        for e in &per_state_fdir[s] {
            fdir_trans_monitor.push(e.mon_idx);
            fdir_trans_target.push(e.target);
            cursor += 1;
        }
        fdir_trans_end[s] = cursor;
    }

    let mut combined = sm.timeline.clone();
    combined.extend(obc.timeline.iter().cloned());
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut tc_times = Vec::with_capacity(combined.len() + 1);
    let mut tc_codes = Vec::with_capacity(combined.len() + 1);
    let mut tc_goals = Vec::with_capacity(combined.len() + 1);
    for (t, cmd, g) in combined {
        tc_times.push(t);
        tc_codes.push(get_cmd_code(&cmd));
        let g_idx = if let Some(goal_arc) = g {
            gt.register_named(Some(goal_arc), "Goal") as i64
        } else {
            -1
        };
        tc_goals.push(g_idx);
    }
    tc_times.push(f64::INFINITY);
    tc_codes.push(-1);
    tc_goals.push(-1);

    let (
        fdir_enabled,
        fdir_safe_state,
        fdir_monitor_kinds,
        fdir_monitor_thresholds,
        fdir_monitor_targets,
    ) = if let Some(ref fdir) = obc.fdir {
        if !fdir.angular_rate_monitors.is_empty() {
            let safe_s = resolve_target(&fdir.safe_mode, &modes_by_name, &leaf_ids)?;
            let mut kinds = Vec::with_capacity(fdir.angular_rate_monitors.len());
            let mut thresholds = Vec::with_capacity(fdir.angular_rate_monitors.len());
            let mut targets = Vec::with_capacity(fdir.angular_rate_monitors.len());
            for mon in &fdir.angular_rate_monitors {
                kinds.push(FDIR_ANGULAR_RATE);
                thresholds.push(mon.max_rate);
                let tgt = match &mon.target_mode {
                    Some(m) => resolve_target(m, &modes_by_name, &leaf_ids)?,
                    None => safe_s,
                };
                targets.push(tgt);
            }
            (true, safe_s, kinds, thresholds, targets)
        } else {
            (false, 0, Vec::new(), Vec::new(), Vec::new())
        }
    } else {
        (false, 0, Vec::new(), Vec::new(), Vec::new())
    };

    Ok(OBCRuntime {
        enabled: true,
        initial_state,
        n_states,
        tc_trans_start,
        tc_trans_end,
        tc_trans_code,
        tc_trans_target,
        tc_trans_goal,
        fov_trans_start,
        fov_trans_end,
        fov_ecef,
        fov_cos_limit,
        fov_negate,
        fov_target,
        fov_goal,
        state_duration,
        state_auto_target,
        state_adcs_mode,
        state_adcs_goal,
        state_sensor_mask,
        state_rw_mask,
        state_mtq_mask,
        tc_trans_sensor_mask,
        tc_trans_rw_mask,
        tc_trans_mtq_mask,
        tc_times,
        tc_codes,
        tc_goals,
        fdir_enabled,
        fdir_safe_state,
        fdir_monitor_kinds,
        fdir_monitor_thresholds,
        fdir_monitor_targets,
        fdir_trans_start,
        fdir_trans_end,
        fdir_trans_monitor,
        fdir_trans_target,
        mode_names: leaf_names,
    })
}

pub fn init_obc_work(runtime: &OBCRuntime) -> [f64; 5] {
    let auto_t = if runtime.enabled && runtime.state_duration[runtime.initial_state] > 0.0 {
        runtime.state_duration[runtime.initial_state]
    } else {
        f64::INFINITY
    };
    [runtime.initial_state as f64, 0.0, auto_t, 0.0, 0.0]
}

#[inline]
fn enter_state(
    target: usize,
    elapsed_s: f64,
    obc: &OBCRuntime,
    work: &mut [f64; 5],
    goal_override: i64,
) -> (usize, i64, i64) {
    work[0] = target as f64;
    let dur = obc.state_duration[target];
    work[2] = if dur > 0.0 {
        elapsed_s + dur
    } else {
        f64::INFINITY
    };
    work[4] = elapsed_s;
    let adcs_m = obc.state_adcs_mode[target];
    let adcs_g = if goal_override >= 0 {
        goal_override
    } else {
        obc.state_adcs_goal[target]
    };
    (target, adcs_m, adcs_g)
}

pub fn step_obc_core(
    y: &[f64],
    elapsed_s: f64,
    epoch_jd: f64,
    dt_obc: f64,
    obc: &OBCRuntime,
    work: &mut [f64; 5],
) -> (usize, i64, i64, i64, i64) {
    let mut state = work[0] as usize;
    let mut cmd_mode = -1_i64;
    let mut cmd_goal = -1_i64;
    let mut entered_state = -1_i64;
    let mut fired_tc_idx = -1_i64;

    if obc.fdir_enabled && work[3] == 0.0 {
        for i in 0..obc.fdir_monitor_kinds.len() {
            if obc.fdir_monitor_kinds[i] == FDIR_ANGULAR_RATE {
                let (wx, wy, wz) = (y[4], y[5], y[6]);
                let limit = obc.fdir_monitor_thresholds[i];
                if wx * wx + wy * wy + wz * wz > limit * limit {
                    work[3] = 1.0;
                    let mut fdir_target = obc.fdir_monitor_targets[i];
                    let f_start = obc.fdir_trans_start[state];
                    let f_end = obc.fdir_trans_end[state];
                    for k in f_start..f_end {
                        let mon_filter = obc.fdir_trans_monitor[k];
                        if mon_filter < 0 || mon_filter == i as i64 {
                            fdir_target = obc.fdir_trans_target[k];
                            break;
                        }
                    }
                    let (st, cm, cg) = enter_state(fdir_target, elapsed_s, obc, work, -1);
                    return (st, cm, cg, st as i64, -1);
                }
            }
        }
    }

    let mut tc_idx = work[1] as usize;
    let next_tc_t = obc.tc_times[tc_idx];
    let auto_t = work[2];
    if elapsed_s >= next_tc_t || elapsed_s >= auto_t {
        while elapsed_s >= obc.tc_times[tc_idx] {
            let cmd = obc.tc_codes[tc_idx];
            let tl_goal = obc.tc_goals[tc_idx];
            tc_idx += 1;
            work[1] = tc_idx as f64;
            let start = obc.tc_trans_start[state];
            let end = obc.tc_trans_end[state];
            for k in start..end {
                if obc.tc_trans_code[k] == cmd {
                    let target = obc.tc_trans_target[k];
                    if target != state {
                        work[3] = 0.0;
                    }
                    let tr_goal = obc.tc_trans_goal[k];
                    let override_goal = if tl_goal >= 0 { tl_goal } else { tr_goal };
                    let (st, cm, cg) = enter_state(target, elapsed_s, obc, work, override_goal);
                    state = st;
                    entered_state = st as i64;
                    fired_tc_idx = k as i64;
                    if cm >= 0 {
                        cmd_mode = cm;
                    }
                    if cg >= 0 {
                        cmd_goal = cg;
                    }
                    break;
                }
            }
        }

        if elapsed_s >= work[2] {
            let auto_target = obc.state_auto_target[state];
            if auto_target >= 0 {
                let tgt = auto_target as usize;
                if tgt != state {
                    work[3] = 0.0;
                }
                let (st, cm, cg) = enter_state(tgt, elapsed_s, obc, work, -1);
                state = st;
                entered_state = st as i64;
                if cm >= 0 {
                    cmd_mode = cm;
                }
                if cg >= 0 {
                    cmd_goal = cg;
                }
            }
        }
    }

    let fov_start = obc.fov_trans_start[state];
    let fov_end = obc.fov_trans_end[state];
    if fov_start < fov_end && elapsed_s >= work[4] {
        let r_sat = [y[7], y[8], y[9]];
        let v_sat = [y[10], y[11], y[12]];
        let r_norm = (r_sat[0] * r_sat[0] + r_sat[1] * r_sat[1] + r_sat[2] * r_sat[2]).sqrt();
        let v_norm = (v_sat[0] * v_sat[0] + v_sat[1] * v_sat[1] + v_sat[2] * v_sat[2]).sqrt();
        let rot = ecef_to_gcrf_matrix_native(epoch_jd, epoch_jd, 0.0, 0.0);
        let mut min_skip = f64::INFINITY;

        for k in fov_start..fov_end {
            let t_ecef = obc.fov_ecef[k];
            let t_eci = rot * Vector3::new(t_ecef[0], t_ecef[1], t_ecef[2]);
            let dx = t_eci[0] - r_sat[0];
            let dy = t_eci[1] - r_sat[1];
            let dz = t_eci[2] - r_sat[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let rt_sq = t_ecef[0] * t_ecef[0] + t_ecef[1] * t_ecef[1] + t_ecef[2] * t_ecef[2];
            let r_horizon = (r_norm * r_norm - rt_sq).max(0.0).sqrt();

            let in_cone = if dist > r_horizon {
                let v_max = v_norm + OMEGA_EARTH * rt_sq.sqrt();
                let dt_skip = (dist - r_horizon) / v_max.max(1.0);
                if 0.9 * dt_skip < min_skip {
                    min_skip = 0.9 * dt_skip;
                }
                false
            } else {
                let cos_off_nadir = -(dx * r_sat[0] + dy * r_sat[1] + dz * r_sat[2]) / (dist * r_norm);
                min_skip = 0.0;
                cos_off_nadir >= obc.fov_cos_limit[k]
            };

            let triggered = if obc.fov_negate[k] { !in_cone } else { in_cone };
            if triggered {
                let fov_target = obc.fov_target[k];
                if fov_target != state {
                    work[3] = 0.0;
                }
                let (st, cm, cg) = enter_state(fov_target, elapsed_s, obc, work, obc.fov_goal[k]);
                state = st;
                entered_state = st as i64;
                if cm >= 0 {
                    cmd_mode = cm;
                }
                if cg >= 0 {
                    cmd_goal = cg;
                }
                return (state, cmd_mode, cmd_goal, entered_state, fired_tc_idx);
            }
        }

        if min_skip > dt_obc && min_skip.is_finite() {
            work[4] = elapsed_s + min_skip;
        } else {
            work[4] = elapsed_s + dt_obc;
        }
    }

    (state, cmd_mode, cmd_goal, entered_state, fired_tc_idx)
}

pub fn step_obc(
    y: &[f64],
    elapsed_s: f64,
    epoch_jd: f64,
    dt_obc: f64,
    obc: &OBCRuntime,
    work: &mut [f64; 5],
) -> (usize, i64, i64) {
    let (state, cmd_mode, cmd_goal, _, _) = step_obc_core(y, elapsed_s, epoch_jd, dt_obc, obc, work);
    (state, cmd_mode, cmd_goal)
}
