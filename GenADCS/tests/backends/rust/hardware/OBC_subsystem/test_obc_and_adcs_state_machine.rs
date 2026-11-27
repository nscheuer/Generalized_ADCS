use std::collections::BTreeMap;
use std::sync::Arc;

use adcs_subsystem::goals::{NadirGoal, QuaternionGoal};
use frames_native::ecef_to_gcrf_matrix_native;
use nalgebra::Vector3;

use crate::fdir::{AngularRateMonitor, FDIR};
use crate::obc_subsystem::OBCSubsystem;
use crate::runtime::{init_obc_work, prepare_obc, step_obc, GoalTable};
use crate::state_machine::{
    ADCSBinding, After, Completion, FDIRTriggered, InFOV, Mode, ModeBindings, StateMachine,
    Telecommand, Transition, Trigger,
};

fn make_state_vec(rates: [f64; 3], pos: [f64; 3], vel: [f64; 3]) -> [f64; 13] {
    [
        1.0, 0.0, 0.0, 0.0, rates[0], rates[1], rates[2], pos[0], pos[1], pos[2], vel[0], vel[1],
        vel[2],
    ]
}

#[test]
fn test_obc_telecommand_same_step_preserves_adcs_binding_and_goal() {
    let mut gt = GoalTable::new();
    let mut adcs_modes = BTreeMap::new();
    adcs_modes.insert("detumble".to_string(), 0);
    adcs_modes.insert("nominal".to_string(), 1);

    let q_override = Arc::new(QuaternionGoal::new([0.0, 1.0, 0.0, 0.0], None).unwrap());
    let sm = StateMachine::new(
        "boot".to_string(),
        vec![
            Mode::new(
                "boot".to_string(),
                vec![],
                None,
                None,
                None,
                ModeBindings {
                    adcs: Some(ADCSBinding::ModeName("detumble".to_string())),
                    ..Default::default()
                },
            )
            .unwrap(),
            Mode::new(
                "nominal".to_string(),
                vec![],
                None,
                None,
                None,
                ModeBindings {
                    adcs: Some(ADCSBinding::ModeAndGoal(
                        "nominal".to_string(),
                        Arc::new(NadirGoal::new()),
                    )),
                    ..Default::default()
                },
            )
            .unwrap(),
            Mode::new("logging".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
        ],
        vec![
            Transition::new(
                "boot".to_string(),
                "nominal".to_string(),
                Trigger::Telecommand(
                    Telecommand::new("GO_NOMINAL".to_string(), Some(q_override), None, None)
                        .unwrap(),
                ),
                0,
            )
            .unwrap(),
            Transition::new(
                "nominal".to_string(),
                "logging".to_string(),
                Trigger::Telecommand(
                    Telecommand::new("START_LOG".to_string(), None, None, None).unwrap(),
                ),
                0,
            )
            .unwrap(),
        ],
        vec![
            (1.0, "GO_NOMINAL".to_string(), None),
            (1.0, "START_LOG".to_string(), None),
        ],
    )
    .unwrap();
    let obc = OBCSubsystem::new(Some(sm), None, 1, vec![]).unwrap();
    let rt = prepare_obc(Some(&obc), &adcs_modes, Some(&mut gt), &[], &[], &[]).unwrap();
    let mut work = init_obc_work(&rt);
    let y = make_state_vec([0.0; 3], [6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);

    let (state, cm, cg) = step_obc(&y, 1.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[state], "logging");
    assert_eq!(cm, 1);
    assert!(cg > 0);
}

#[test]
fn test_obc_after_and_completion_transitions() {
    let mut gt = GoalTable::new();
    let mut adcs_modes = BTreeMap::new();
    adcs_modes.insert("idle".to_string(), 0);

    let sm = StateMachine::new(
        "warmup".to_string(),
        vec![
            Mode::new("warmup".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
            Mode::new("work".to_string(), vec![], None, Some(3.0), None, ModeBindings::default())
                .unwrap(),
            Mode::new("done".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
        ],
        vec![
            Transition::new(
                "warmup".to_string(),
                "work".to_string(),
                Trigger::After(After::new(2.0).unwrap()),
                0,
            )
            .unwrap(),
            Transition::new(
                "work".to_string(),
                "done".to_string(),
                Trigger::Completion(Completion),
                0,
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let obc = OBCSubsystem::new(Some(sm), None, 1, vec![]).unwrap();
    let rt = prepare_obc(Some(&obc), &adcs_modes, Some(&mut gt), &[], &[], &[]).unwrap();
    let mut work = init_obc_work(&rt);
    let y = make_state_vec([0.0; 3], [6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);

    let (s0, _, _) = step_obc(&y, 0.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s0], "warmup");
    let (s1, _, _) = step_obc(&y, 2.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s1], "work");
    let (s2, _, _) = step_obc(&y, 4.9, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s2], "work");
    let (s3, _, _) = step_obc(&y, 5.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s3], "done");
}

#[test]
fn test_obc_infov_entry_and_exit() {
    let jd = 2_461_000.5;
    let r_eci = Vector3::new(6_878_137.0, 0.0, 0.0);
    let r_ecef = ecef_to_gcrf_matrix_native(jd, jd, 0.0, 0.0).transpose() * r_eci;
    let subsat_lon = r_ecef[1].atan2(r_ecef[0]).to_degrees();
    let swath = InFOV::new(0.0, subsat_lon, 0.0, 15.0, false, None).unwrap();
    let swath_exit = swath.inverted();

    let mut gt = GoalTable::new();
    let mut adcs_modes = BTreeMap::new();
    adcs_modes.insert("idle".to_string(), 0);
    let sm = StateMachine::new(
        "idle".to_string(),
        vec![
            Mode::new("idle".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
            Mode::new("payload".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
        ],
        vec![
            Transition::new("idle".to_string(), "payload".to_string(), Trigger::InFOV(swath), 0)
                .unwrap(),
            Transition::new(
                "payload".to_string(),
                "idle".to_string(),
                Trigger::InFOV(swath_exit),
                0,
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let obc = OBCSubsystem::new(Some(sm), None, 1, vec![]).unwrap();
    let rt = prepare_obc(Some(&obc), &adcs_modes, Some(&mut gt), &[], &[], &[]).unwrap();
    let mut work = init_obc_work(&rt);

    let y_over = make_state_vec([0.0; 3], [6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);
    let (s_in, _, cg) = step_obc(&y_over, 0.0, jd, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s_in], "payload");
    assert!(gt.names[cg as usize].contains("CoordinateGoal"));

    let y_far = make_state_vec([0.0; 3], [-6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);
    let (s_out, _, _) = step_obc(&y_far, 1.0, jd, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s_out], "idle");
}

#[test]
fn test_obc_fdir_triggered_custom_target_and_unlatch() {
    let mut gt = GoalTable::new();
    let mut adcs_modes = BTreeMap::new();
    adcs_modes.insert("nominal".to_string(), 0);
    adcs_modes.insert("safe".to_string(), 1);

    let sm = StateMachine::new(
        "nominal".to_string(),
        vec![
            Mode::new(
                "nominal".to_string(),
                vec![],
                None,
                None,
                None,
                ModeBindings {
                    adcs: Some(ADCSBinding::ModeName("nominal".to_string())),
                    ..Default::default()
                },
            )
            .unwrap(),
            Mode::new(
                "alt_safe".to_string(),
                vec![],
                None,
                None,
                None,
                ModeBindings {
                    adcs: Some(ADCSBinding::ModeName("safe".to_string())),
                    ..Default::default()
                },
            )
            .unwrap(),
            Mode::new(
                "safe_mode".to_string(),
                vec![],
                None,
                None,
                None,
                ModeBindings {
                    adcs: Some(ADCSBinding::ModeName("safe".to_string())),
                    ..Default::default()
                },
            )
            .unwrap(),
        ],
        vec![
            Transition::new(
                "nominal".to_string(),
                "alt_safe".to_string(),
                Trigger::FDIRTriggered(FDIRTriggered {
                    monitor: Some("rate_mon".to_string()),
                }),
                0,
            )
            .unwrap(),
            Transition::new(
                "alt_safe".to_string(),
                "nominal".to_string(),
                Trigger::After(After::new(2.0).unwrap()),
                0,
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let fdir = FDIR::new(
        vec![AngularRateMonitor::new(0.05, Some("rate_mon".to_string()), None).unwrap()],
        None,
    )
    .unwrap();
    let obc = OBCSubsystem::new(Some(sm), Some(fdir), 1, vec![]).unwrap();
    let rt = prepare_obc(Some(&obc), &adcs_modes, Some(&mut gt), &[], &[], &[]).unwrap();
    let mut work = init_obc_work(&rt);

    let y_fast = make_state_vec([0.1, 0.0, 0.0], [6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);
    let (s1, cm1, _) = step_obc(&y_fast, 0.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s1], "alt_safe");
    assert_eq!(cm1, 1);
    assert_eq!(work[3], 1.0);

    let y_calm = make_state_vec([0.0; 3], [6_878_137.0, 0.0, 0.0], [0.0, 7_612.6, 0.0]);
    let (s2, cm2, _) = step_obc(&y_calm, 2.0, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s2], "nominal");
    assert_eq!(cm2, 0);
    assert_eq!(work[3], 0.0);

    let (s3, _, _) = step_obc(&y_fast, 2.5, 2_461_000.5, 0.5, &rt, &mut work);
    assert_eq!(rt.mode_names[s3], "alt_safe");
    assert_eq!(work[3], 1.0);
}

#[test]
fn test_obc_fdir_triggered_unknown_monitor_raises() {
    let sm = StateMachine::new(
        "nominal".to_string(),
        vec![
            Mode::new("nominal".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
            Mode::new("safe_mode".to_string(), vec![], None, None, None, ModeBindings::default())
                .unwrap(),
        ],
        vec![Transition::new(
            "nominal".to_string(),
            "safe_mode".to_string(),
            Trigger::FDIRTriggered(FDIRTriggered {
                monitor: Some("missing".to_string()),
            }),
            0,
        )
        .unwrap()],
        vec![],
    )
    .unwrap();
    let fdir = FDIR::new(
        vec![AngularRateMonitor::new(0.05, Some("rate_mon".to_string()), None).unwrap()],
        None,
    )
    .unwrap();
    let obc = OBCSubsystem::new(Some(sm), Some(fdir), 1, vec![]).unwrap();
    assert!(prepare_obc(Some(&obc), &BTreeMap::new(), None, &[], &[], &[]).is_err());
}
