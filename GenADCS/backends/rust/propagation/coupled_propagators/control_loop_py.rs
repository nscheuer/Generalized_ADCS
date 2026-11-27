use std::collections::BTreeMap;
use std::sync::Arc;

use adcs_subsystem::controllers::allocator::{Allocator, L2Allocator, LInfinityAllocator};
use adcs_subsystem::controllers::attitude_controller::AttitudeController;
use adcs_subsystem::controllers::bcross::BCross;
use adcs_subsystem::controllers::bdot::BDot;
use adcs_subsystem::controllers::controller::{Controller, NoController};
use adcs_subsystem::controllers::desaturation::{Desaturation, HoganSchaubManagement, NoDesaturation};
use adcs_subsystem::controllers::feedback::{Feedback, GainInput, MRPFeedbackLinearized, Mat3};
use adcs_subsystem::controllers::feedforward::{Feedforward, Passthrough};
use adcs_subsystem::controllers::runtime::{prepare_controllers, ControllerTable};
use adcs_subsystem::goals::{
    AlongTrackGoal, AngularRateGoal, CoordinateFMCGoal, CoordinateGoal, CrossTrackGoal, Goal,
    NadirGoal, NoGoal, QuaternionGoal, SunGoal, TwoVectorGoal, VectorGoal, VelocityGoal,
    ZenithGoal,
};
use adcs_subsystem::state_machine::{
    prepare_adcs_sm, ADCSMode, ADCSStateMachine, ADCSTimelineTarget, DeviceMaskSpec, GoalTable as AdcsGoalTable,
    GOAL_ALONG_TRACK, GOAL_ANGULAR_RATE, GOAL_COORDINATE, GOAL_COORDINATE_FMC, GOAL_CROSS_TRACK,
    GOAL_NADIR, GOAL_NONE, GOAL_PARAM_WIDTH, GOAL_QUATERNION, GOAL_SUN, GOAL_VELOCITY, GOAL_ZENITH,
};
use numpy::IntoPyArray;
use obc_subsystem::fdir::{AngularRateMonitor, FDIRMonitor, FDIR};
use obc_subsystem::obc_subsystem::OBCSubsystem;
use obc_subsystem::runtime::{prepare_obc, GoalTable as ObcGoalTable};
use obc_subsystem::state_machine::{
    ADCSBinding, After, Completion, FDIRTriggered, HardwareMaskSpec, InFOV, Mode, ModeBindings,
    StateMachine, Telecommand, TimelineEntry, Transition, Trigger,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::control_loop::ControlLoopContext;
use super::simulation_sampling::SimulationSampler;
use super::Runtime;

fn vec3_from_py(obj: &PyAny) -> PyResult<[f64; 3]> {
    let v: Vec<f64> = if obj.hasattr("tolist")? {
        obj.call_method0("tolist")?.extract()?
    } else {
        obj.extract()?
    };
    if v.len() != 3 {
        return Err(PyValueError::new_err("expected 3-element vector"));
    }
    Ok([v[0], v[1], v[2]])
}

fn vec4_from_py(obj: &PyAny) -> PyResult<[f64; 4]> {
    let v: Vec<f64> = if obj.hasattr("tolist")? {
        obj.call_method0("tolist")?.extract()?
    } else {
        obj.extract()?
    };
    if v.len() != 4 {
        return Err(PyValueError::new_err("expected 4-element vector"));
    }
    Ok([v[0], v[1], v[2], v[3]])
}

fn mat3_from_py(obj: &PyAny) -> PyResult<Mat3> {
    let m: Vec<Vec<f64>> = if obj.hasattr("tolist")? {
        obj.call_method0("tolist")?.extract()?
    } else {
        obj.extract()?
    };
    if m.len() != 3 || m.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err("expected 3x3 matrix"));
    }
    Ok([
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ])
}

fn extract_goal(
    obj: &PyAny,
) -> PyResult<(Arc<dyn Goal + Send + Sync>, i32, [f64; GOAL_PARAM_WIDTH], String)> {
    if obj.is_none() {
        return Ok((Arc::new(NoGoal), GOAL_NONE, [0.0; 4], "NoGoal".to_string()));
    }
    let name = obj.get_type().name()?;
    match name {
        "NoGoal" => Ok((Arc::new(NoGoal), GOAL_NONE, [0.0; 4], "NoGoal".to_string())),
        "NadirGoal" => Ok((Arc::new(NadirGoal), GOAL_NADIR, [0.0; 4], "NadirGoal".to_string())),
        "ZenithGoal" => Ok((Arc::new(ZenithGoal), GOAL_ZENITH, [0.0; 4], "ZenithGoal".to_string())),
        "AlongTrackGoal" => Ok((
            Arc::new(AlongTrackGoal),
            GOAL_ALONG_TRACK,
            [0.0; 4],
            "AlongTrackGoal".to_string(),
        )),
        "CrossTrackGoal" => Ok((
            Arc::new(CrossTrackGoal),
            GOAL_CROSS_TRACK,
            [0.0; 4],
            "CrossTrackGoal".to_string(),
        )),
        "VelocityGoal" => Ok((
            Arc::new(VelocityGoal),
            GOAL_VELOCITY,
            [0.0; 4],
            "VelocityGoal".to_string(),
        )),
        "SunGoal" => {
            let g = SunGoal::new().map_err(PyValueError::new_err)?;
            Ok((Arc::new(g), GOAL_SUN, [0.0; 4], "SunGoal".to_string()))
        }
        "CoordinateGoal" => {
            let lat: f64 = obj.getattr("lat_deg")?.extract()?;
            let lon: f64 = obj.getattr("lon_deg")?.extract()?;
            let alt: f64 = obj.getattr("alt_km")?.extract()?;
            let g = CoordinateGoal::new(lat, lon, Some(alt)).map_err(PyValueError::new_err)?;
            Ok((
                Arc::new(g),
                GOAL_COORDINATE,
                [lat, lon, alt, 0.0],
                "CoordinateGoal".to_string(),
            ))
        }
        "CoordinateFMCGoal" => {
            let lat: f64 = obj.getattr("lat_deg")?.extract()?;
            let lon: f64 = obj.getattr("lon_deg")?.extract()?;
            let alt: f64 = obj.getattr("alt_km")?.extract()?;
            let fmc: i64 = obj.getattr("fmc_level")?.extract()?;
            let g = CoordinateFMCGoal::new(lat, lon, Some(alt), Some(fmc))
                .map_err(PyValueError::new_err)?;
            Ok((
                Arc::new(g),
                GOAL_COORDINATE_FMC,
                [lat, lon, alt, fmc as f64],
                "CoordinateFMCGoal".to_string(),
            ))
        }
        "QuaternionGoal" => {
            let q = vec4_from_py(obj.getattr("quaternion")?)?;
            let rate = if obj.hasattr("rate")? && !obj.getattr("rate")?.is_none() {
                Some(vec3_from_py(obj.getattr("rate")?)?)
            } else {
                None
            };
            let g = QuaternionGoal::new(q, rate).map_err(PyValueError::new_err)?;
            Ok((
                Arc::new(g),
                GOAL_QUATERNION,
                q,
                "QuaternionGoal".to_string(),
            ))
        }
        "AngularRateGoal" => {
            let w = vec3_from_py(obj.getattr("rate")?)?;
            let g = AngularRateGoal::new(w).map_err(PyValueError::new_err)?;
            Ok((
                Arc::new(g),
                GOAL_ANGULAR_RATE,
                [w[0], w[1], w[2], 0.0],
                "AngularRateGoal".to_string(),
            ))
        }
        "VectorGoal" => {
            let iv = vec3_from_py(obj.getattr("inertial_vector")?)?;
            let bv = vec3_from_py(obj.getattr("body_vector")?)?;
            let g = VectorGoal::new(iv, Some(bv), None, None).map_err(PyValueError::new_err)?;
            Ok((Arc::new(g), GOAL_NONE, [0.0; 4], "VectorGoal".to_string()))
        }
        "TwoVectorGoal" => {
            let iv1 = vec3_from_py(obj.getattr("primary_inertial")?)?;
            let iv2 = vec3_from_py(obj.getattr("secondary_inertial")?)?;
            let bv1 = vec3_from_py(obj.getattr("primary_body")?)?;
            let bv2 = vec3_from_py(obj.getattr("secondary_body")?)?;
            let g = TwoVectorGoal::new([iv1, iv2], [bv1, bv2]).map_err(PyValueError::new_err)?;
            Ok((Arc::new(g), GOAL_NONE, [0.0; 4], "TwoVectorGoal".to_string()))
        }
        other => Err(PyValueError::new_err(format!("unsupported goal type: {other}"))),
    }
}

fn gain_input_from_py(obj: &PyAny) -> PyResult<GainInput> {
    if let Ok(s) = obj.extract::<f64>() {
        return Ok(GainInput::Scalar(s));
    }
    let list_obj = if obj.hasattr("tolist")? {
        obj.call_method0("tolist")?
    } else {
        obj
    };
    if let Ok(v) = list_obj.extract::<Vec<f64>>() {
        if v.len() == 3 {
            return Ok(GainInput::Vector([v[0], v[1], v[2]]));
        }
    }
    if let Ok(m) = list_obj.extract::<Vec<Vec<f64>>>() {
        if m.len() == 3 && m.iter().all(|r| r.len() == 3) {
            return Ok(GainInput::Matrix([
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ]));
        }
    }
    Err(PyValueError::new_err("invalid gain input"))
}

fn extract_controller(obj: &PyAny) -> PyResult<Option<Arc<dyn Controller>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let name = obj.get_type().name()?;
    match name {
        "NoController" => Ok(Some(Arc::new(NoController))),
        "BDot" => {
            let gain: f64 = obj.getattr("gain")?.extract()?;
            Ok(Some(Arc::new(
                BDot::new(gain).map_err(PyValueError::new_err)?,
            )))
        }
        "BCross" => {
            let gain: f64 = obj.getattr("gain")?.extract()?;
            Ok(Some(Arc::new(
                BCross::new(gain).map_err(PyValueError::new_err)?,
            )))
        }
        "AttitudeController" => {
            let fb_obj = obj.getattr("feedback")?;
            let fb_name = fb_obj.get_type().name()?;
            let feedback: Arc<dyn Feedback> = match fb_name {
                "MRPFeedbackLinearized" => {
                    let kp = gain_input_from_py(fb_obj.getattr("kp")?)?;
                    let kd = gain_input_from_py(fb_obj.getattr("kd")?)?;
                    Arc::new(MRPFeedbackLinearized::new(kp, kd).map_err(PyValueError::new_err)?)
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported feedback type: {fb_name}"
                    )))
                }
            };
            let ff_obj = obj.getattr("feedforward")?;
            let feedforward: Option<Arc<dyn Feedforward>> = if ff_obj.is_none() {
                None
            } else {
                Some(Arc::new(Passthrough))
            };
            let alloc_obj = obj.getattr("allocator")?;
            let allocator: Option<Arc<dyn Allocator>> = if alloc_obj.is_none() {
                None
            } else {
                match alloc_obj.get_type().name()? {
                    "L2Allocator" => Some(Arc::new(L2Allocator)),
                    "LInfinityAllocator" => Some(Arc::new(LInfinityAllocator)),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported allocator type: {other}"
                        )))
                    }
                }
            };
            let desat_obj = obj.getattr("desaturation")?;
            let desaturation: Option<Arc<dyn Desaturation>> = if desat_obj.is_none() {
                None
            } else {
                match desat_obj.get_type().name()? {
                    "NoDesaturation" => Some(Arc::new(NoDesaturation)),
                    "HoganSchaubManagement" => {
                        let gain: f64 = desat_obj.getattr("gain")?.extract()?;
                        let target_momentum: f64 =
                            desat_obj.getattr("target_momentum")?.extract()?;
                        let mtq_gain: f64 = desat_obj.getattr("mtq_gain")?.extract()?;
                        Some(Arc::new(
                            HoganSchaubManagement::new(gain, target_momentum, mtq_gain)
                                .map_err(PyValueError::new_err)?,
                        ))
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported desaturation type: {other}"
                        )))
                    }
                }
            };
            Ok(Some(Arc::new(AttitudeController::new(
                feedback,
                feedforward,
                allocator,
                desaturation,
            ))))
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported controller type: {other}"
        ))),
    }
}

fn parse_device_mask(obj: &PyAny) -> PyResult<DeviceMaskSpec> {
    let mut map = BTreeMap::new();
    if obj.is_none() {
        return Ok(map);
    }
    let dict = obj.downcast::<PyDict>()?;
    for (k, v) in dict.iter() {
        map.insert(k.extract::<String>()?, v.extract::<bool>()?);
    }
    Ok(map)
}

fn parse_opt_mask(obj: &PyAny) -> PyResult<Option<HardwareMaskSpec>> {
    if obj.is_none() {
        return Ok(None);
    }
    Ok(Some(parse_device_mask(obj)?))
}

fn extract_adcs_sm(obj: &PyAny) -> PyResult<ADCSStateMachine> {
    let initial: String = obj.getattr("initial")?.extract()?;
    let mut modes = Vec::new();
    for item in obj.getattr("modes")?.iter()? {
        let m = item?;
        let name: String = m.getattr("name")?.extract()?;
        let (goal_arc, goal_kind, goal_params, goal_label) = extract_goal(m.getattr("goal")?)?;
        let ctrl_arc = extract_controller(m.getattr("controller")?)?;
        let sensors = parse_device_mask(m.getattr("sensors")?)?;
        let actuators = parse_device_mask(m.getattr("actuators")?)?;
        modes.push(
            ADCSMode::new(
                name,
                Some(goal_arc),
                goal_kind,
                goal_params,
                goal_label,
                ctrl_arc,
                sensors,
                actuators,
            )
            .map_err(PyValueError::new_err)?,
        );
    }
    let mut timeline = Vec::new();
    let tl_obj = obj.getattr("timeline")?;
    if !tl_obj.is_none() {
        for item in tl_obj.iter()? {
            let tup = item?.downcast::<PyTuple>()?;
            let t: f64 = tup.get_item(0)?.extract()?;
            let target = tup.get_item(1)?;
            if let Ok(m_name) = target.extract::<String>() {
                timeline.push((t, ADCSTimelineTarget::ModeName(m_name)));
            } else {
                let (goal, kind, params, label) = extract_goal(target)?;
                timeline.push((
                    t,
                    ADCSTimelineTarget::GoalSpec {
                        goal,
                        kind,
                        params,
                        label,
                    },
                ));
            }
        }
    }
    ADCSStateMachine::new(initial, modes, timeline).map_err(PyValueError::new_err)
}

fn extract_obc_mode(obj: &PyAny) -> PyResult<Mode> {
    let name: String = obj.getattr("name")?.extract()?;
    let initial_submode: Option<String> = obj.getattr("initial_submode")?.extract()?;
    let duration: Option<f64> = obj.getattr("duration")?.extract()?;
    let auto_target: Option<String> = obj.getattr("auto_target")?.extract()?;
    let mut submodes = Vec::new();
    for item in obj.getattr("submodes")?.iter()? {
        submodes.push(extract_obc_mode(item?)?);
    }
    let mut bindings = ModeBindings::default();
    let b_obj = obj.getattr("bindings")?;
    if !b_obj.is_none() {
        let d = b_obj.downcast::<PyDict>()?;
        if let Some(adcs_val) = d.get_item("adcs")? {
            if let Ok(s) = adcs_val.extract::<String>() {
                bindings.adcs = Some(ADCSBinding::ModeName(s));
            } else if let Ok(tup) = adcs_val.downcast::<PyTuple>() {
                let m_name: String = tup.get_item(0)?.extract()?;
                let (g, _, _, _) = extract_goal(tup.get_item(1)?)?;
                bindings.adcs = Some(ADCSBinding::ModeAndGoal(m_name, g));
            } else if !adcs_val.is_none() {
                let (g, _, _, _) = extract_goal(adcs_val)?;
                bindings.adcs = Some(ADCSBinding::GoalOnly(g));
            }
        }
        if let Some(s_val) = d.get_item("sensors")? {
            bindings.sensors = parse_opt_mask(s_val)?;
        }
        if let Some(a_val) = d.get_item("actuators")? {
            bindings.actuators = parse_opt_mask(a_val)?;
        }
    }
    Mode::new(name, submodes, initial_submode, duration, auto_target, bindings)
        .map_err(PyValueError::new_err)
}

fn extract_obc_timeline(obj: &PyAny) -> PyResult<Vec<TimelineEntry>> {
    let mut out = Vec::new();
    if obj.is_none() {
        return Ok(out);
    }
    for item in obj.iter()? {
        let tup = item?.downcast::<PyTuple>()?;
        let t: f64 = tup.get_item(0)?.extract()?;
        let cmd: String = tup.get_item(1)?.extract()?;
        let g = if tup.len() >= 3 && !tup.get_item(2)?.is_none() {
            Some(extract_goal(tup.get_item(2)?)?.0)
        } else {
            None
        };
        out.push((t, cmd, g));
    }
    Ok(out)
}

fn extract_obc_subsystem(obj: &PyAny) -> PyResult<OBCSubsystem> {
    let sm_obj = obj.getattr("state_machine")?;
    let sm = if sm_obj.is_none() {
        None
    } else {
        let initial: String = sm_obj.getattr("initial")?.extract()?;
        let mut modes = Vec::new();
        for item in sm_obj.getattr("modes")?.iter()? {
            modes.push(extract_obc_mode(item?)?);
        }
        let mut transitions = Vec::new();
        for item in sm_obj.getattr("transitions")?.iter()? {
            let tr = item?;
            let source: String = tr.getattr("source")?.extract()?;
            let target: String = tr.getattr("target")?.extract()?;
            let priority: i32 = tr.getattr("priority")?.extract()?;
            let trig_obj = tr.getattr("trigger")?;
            let trig = match trig_obj.get_type().name()? {
                "Telecommand" => {
                    let command: String = trig_obj.getattr("command")?.extract()?;
                    let g_obj = trig_obj.getattr("goal")?;
                    let goal = if g_obj.is_none() {
                        None
                    } else {
                        Some(extract_goal(g_obj)?.0)
                    };
                    let sensors = parse_opt_mask(trig_obj.getattr("sensors")?)?;
                    let actuators = parse_opt_mask(trig_obj.getattr("actuators")?)?;
                    Trigger::Telecommand(
                        Telecommand::new(command, goal, sensors, actuators)
                            .map_err(PyValueError::new_err)?,
                    )
                }
                "InFOV" => {
                    let lat: f64 = trig_obj.getattr("lat")?.extract()?;
                    let lon: f64 = trig_obj.getattr("lon")?.extract()?;
                    let alt: f64 = trig_obj.getattr("alt")?.extract()?;
                    let max_off: f64 = trig_obj.getattr("max_off_nadir_deg")?.extract()?;
                    let negate: bool = trig_obj.getattr("negate")?.extract()?;
                    let g_obj = trig_obj.getattr("goal")?;
                    let goal = if g_obj.is_none() {
                        None
                    } else {
                        Some(extract_goal(g_obj)?.0)
                    };
                    Trigger::InFOV(
                        InFOV::new(lat, lon, alt, max_off, negate, goal)
                            .map_err(PyValueError::new_err)?,
                    )
                }
                "Completion" => Trigger::Completion(Completion),
                "After" => {
                    let dur: f64 = trig_obj.getattr("duration")?.extract()?;
                    Trigger::After(After::new(dur).map_err(PyValueError::new_err)?)
                }
                "FDIRTriggered" => {
                    let monitor: Option<String> = trig_obj.getattr("monitor")?.extract()?;
                    Trigger::FDIRTriggered(FDIRTriggered { monitor })
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported trigger type: {other}"
                    )))
                }
            };
            transitions.push(Transition::new(source, target, trig, priority).map_err(PyValueError::new_err)?);
        }
        let tl = extract_obc_timeline(sm_obj.getattr("timeline")?)?;
        Some(StateMachine::new(initial, modes, transitions, tl).map_err(PyValueError::new_err)?)
    };

    let fdir_obj = obj.getattr("fdir")?;
    let fdir = if fdir_obj.is_none() {
        None
    } else {
        let safe_mode: String = fdir_obj.getattr("safe_mode")?.extract()?;
        let mut mons: Vec<AngularRateMonitor> = Vec::new();
        for item in fdir_obj.getattr("monitors")?.iter()? {
            let m = item?;
            match m.get_type().name()? {
                "AngularRateMonitor" => {
                    let threshold: f64 = m.getattr("max_rate")?.extract()?;
                    let name: String = m.getattr("name")?.extract()?;
                    let target_mode: Option<String> = m.getattr("target_mode")?.extract()?;
                    mons.push(
                        AngularRateMonitor::new(threshold, Some(name), target_mode)
                            .map_err(PyValueError::new_err)?,
                    );
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported FDIR monitor type: {other}"
                    )))
                }
            }
        }
        Some(FDIR::new(mons, Some(safe_mode)).map_err(PyValueError::new_err)?)
    };

    let n_val: usize = obj.getattr("N")?.extract()?;
    let tl = extract_obc_timeline(obj.getattr("timeline")?)?;
    OBCSubsystem::new(sm, fdir, n_val, tl).map_err(PyValueError::new_err)
}

pub(super) fn build_control_loop_context(
    _py: Python<'_>,
    sampler: &SimulationSampler,
    runtime: &Runtime,
    adcs_opt: Option<&PyAny>,
    obc_opt: Option<&PyAny>,
    epoch_jd: f64,
    dt_ctrl: f64,
    ctrl_log_flags: u32,
    mu: f64,
) -> PyResult<Option<ControlLoopContext>> {
    let adcs = match adcs_opt.filter(|o| !o.is_none()) {
        Some(a) => a,
        None => {
            if obc_opt.filter(|o| !o.is_none()).is_none() {
                return Ok(None);
            }
            return Err(PyValueError::new_err("obc requires adcs"));
        }
    };

    let mut sensor_names: Vec<String> = Vec::new();
    let mut sensor_types: Vec<String> = Vec::new();
    let sensors_py = adcs.getattr("sensors")?;
    if !sensors_py.is_none() {
        for item in sensors_py.iter()? {
            let s = item?;
            let stype = s.get_type().name()?.to_string();
            let sname = if s.hasattr("name")? {
                s.getattr("name")?
                    .extract::<Option<String>>()?
                    .unwrap_or_else(|| stype.clone())
            } else {
                stype.clone()
            };
            sensor_names.push(sname);
            sensor_types.push(stype);
        }
    }
    let sensor_name_refs: Vec<&str> = sensor_names.iter().map(String::as_str).collect();
    let sensor_type_refs: Vec<&str> = sensor_types.iter().map(String::as_str).collect();

    let mut rw_names: Vec<String> = Vec::new();
    let mut rw_axes: Vec<[f64; 3]> = Vec::new();
    let mut rw_max_torque: Vec<f64> = Vec::new();
    let rws_py = adcs.getattr("reaction_wheels")?;
    if !rws_py.is_none() {
        for item in rws_py.iter()? {
            let rw = item?;
            let name = if rw.hasattr("name")? {
                rw.getattr("name")?
                    .extract::<Option<String>>()?
                    .unwrap_or_default()
            } else {
                String::new()
            };
            rw_names.push(name);
            rw_axes.push(vec3_from_py(rw.getattr("orientation")?)?);
            rw_max_torque.push(rw.getattr("max_torque")?.extract::<f64>()?);
        }
    }
    let rw_name_refs: Vec<&str> = rw_names.iter().map(String::as_str).collect();

    let mut mtq_names: Vec<String> = Vec::new();
    let mtqs_py = adcs.getattr("magnetorquers")?;
    if !mtqs_py.is_none() {
        for item in mtqs_py.iter()? {
            let mtq = item?;
            let name = if mtq.hasattr("name")? {
                mtq.getattr("name")?
                    .extract::<Option<String>>()?
                    .unwrap_or_default()
            } else {
                String::new()
            };
            mtq_names.push(name);
        }
    }
    let mtq_name_refs: Vec<&str> = mtq_names.iter().map(String::as_str).collect();

    let sm_obj = adcs.getattr("state_machine")?;
    let has_adcs_sm = !sm_obj.is_none();
    let ctrl_obj = adcs.getattr("controller")?;
    let mut has_ctrl = !ctrl_obj.is_none();
    let obc_obj = obc_opt.filter(|o| !o.is_none());
    let has_obc = obc_obj.is_some();

    if !has_adcs_sm && !has_ctrl && !has_obc {
        return Ok(None);
    }

    let mut adcs_gt = AdcsGoalTable::new();
    let mut ctrl_table = ControllerTable::new();
    let adcs_sm_native = if has_adcs_sm {
        Some(extract_adcs_sm(sm_obj)?)
    } else {
        None
    };
    let mut adcs_sm_rt = prepare_adcs_sm(
        adcs_sm_native.as_ref(),
        Some(&mut adcs_gt),
        Some(&mut ctrl_table),
        &sensor_name_refs,
        &rw_name_refs,
        &mtq_name_refs,
    )
    .map_err(PyValueError::new_err)?;

    let mut obc_gt = ObcGoalTable::new();
    for i in 1..adcs_sm_rt.goal_objects.len() {
        obc_gt.register_named(Some(adcs_sm_rt.goal_objects[i].clone()), "Goal");
    }
    let mut adcs_mode_to_id = BTreeMap::new();
    for (idx, name) in adcs_sm_rt.mode_names.iter().enumerate() {
        adcs_mode_to_id.insert(name.clone(), idx);
    }
    let obc_native = match obc_obj {
        Some(o) => Some(extract_obc_subsystem(o)?),
        None => None,
    };
    let obc_rt = prepare_obc(
        obc_native.as_ref(),
        &adcs_mode_to_id,
        Some(&mut obc_gt),
        &sensor_name_refs,
        &rw_name_refs,
        &mtq_name_refs,
    )
    .map_err(PyValueError::new_err)?;
    for obj in obc_gt.objects.into_iter().skip(adcs_sm_rt.goal_objects.len()) {
        adcs_sm_rt.goal_objects.push(obj);
    }

    let mut ctrl_list: Vec<Option<Arc<dyn Controller>>> = Vec::new();
    if has_adcs_sm {
        // ControllerTable already populated by prepare_adcs_sm; rebuild list from table.
        ctrl_list = ctrl_table.objects.iter().cloned().map(Some).collect();
        if ctrl_list.len() > 1 {
            has_ctrl = true;
        }
    } else if has_ctrl {
        ctrl_list.push(extract_controller(ctrl_obj)?);
    }

    let mtq_axes: Vec<[f64; 3]> = runtime.mtq_axes.clone();
    let mtq_max_dipole: Vec<f64> = runtime.mtq_max_dipole.clone();
    let total_inertia = runtime.inertia;
    let bus_inertia = if adcs.hasattr("inertia")? {
        mat3_from_py(adcs.getattr("inertia")?)?
    } else {
        let mut bi = total_inertia;
        for (axis, &iw) in runtime.wheel_axes.iter().zip(runtime.wheel_inertias.iter()) {
            for r in 0..3 {
                for c in 0..3 {
                    bi[r][c] -= iw * axis[r] * axis[c];
                }
            }
        }
        bi
    };

    let ctrl_rt = prepare_controllers(
        &ctrl_list,
        &sensor_type_refs,
        mtq_axes,
        mtq_max_dipole,
        Some(&mut ctrl_table),
        if rw_axes.is_empty() { None } else { Some(rw_axes) },
        if rw_max_torque.is_empty() { None } else { Some(rw_max_torque) },
        Some(total_inertia),
        Some(bus_inertia),
    )
    .map_err(PyValueError::new_err)?;

    Ok(Some(ControlLoopContext::new(
        has_obc,
        obc_rt,
        has_adcs_sm,
        adcs_sm_rt,
        has_ctrl,
        ctrl_rt,
        sampler,
        mu,
        epoch_jd,
        dt_ctrl,
        ctrl_log_flags,
        rw_names.len(),
        mtq_names.len(),
    )))
}

pub(super) fn attach_control_logs_to_pydict(
    py: Python<'_>,
    samples_dict: &PyAny,
    ctx: &ControlLoopContext,
) -> PyResult<()> {
    let dict = samples_dict.downcast::<PyDict>()?;
    let rows = ctx.recorded_rows;
    if ctx.has_obc || ctx.has_adcs_sm {
        let trace = numpy::ndarray::Array2::from_shape_vec((rows, 3), ctx.sm_trace.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(
            "obc_mode",
            trace
                .slice(numpy::ndarray::s![.., 0..1])
                .to_owned()
                .into_pyarray(py),
        )?;
        dict.set_item(
            "adcs_mode",
            trace
                .slice(numpy::ndarray::s![.., 1..2])
                .to_owned()
                .into_pyarray(py),
        )?;
        dict.set_item(
            "adcs_goal",
            trace
                .slice(numpy::ndarray::s![.., 2..3])
                .to_owned()
                .into_pyarray(py),
        )?;
    }
    if ctx.ctrl_log_flags != 0 {
        let log = numpy::ndarray::Array2::from_shape_vec((rows, 20), ctx.ctrl_log.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if (ctx.ctrl_log_flags & 1) != 0 {
            dict.set_item(
                "goal_attitude",
                log.slice(numpy::ndarray::s![.., 0..4])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
        if (ctx.ctrl_log_flags & 2) != 0 {
            dict.set_item(
                "goal_rate",
                log.slice(numpy::ndarray::s![.., 4..7])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
        if (ctx.ctrl_log_flags & 4) != 0 {
            dict.set_item(
                "feedforward_torque",
                log.slice(numpy::ndarray::s![.., 7..10])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
        if (ctx.ctrl_log_flags & 8) != 0 {
            dict.set_item(
                "feedforward_attitude",
                log.slice(numpy::ndarray::s![.., 10..14])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
        if (ctx.ctrl_log_flags & 16) != 0 {
            dict.set_item(
                "feedforward_rate",
                log.slice(numpy::ndarray::s![.., 14..17])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
        if (ctx.ctrl_log_flags & 32) != 0 {
            dict.set_item(
                "reference_torque",
                log.slice(numpy::ndarray::s![.., 17..20])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(())
}
