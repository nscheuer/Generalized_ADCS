use std::collections::BTreeMap;
use std::sync::Arc;

use crate::controllers::{BDot, ControllerTable};
use crate::goals::{NadirGoal, NoGoal, QuaternionGoal};
use crate::state_machine::{
    evaluate_goal_default_mu, init_adcs_work, prepare_adcs_sm, step_adcs_sm, ADCSMode,
    ADCSStateMachine, ADCSTimelineTarget, GoalTable, GOAL_NADIR, GOAL_NONE, GOAL_PARAM_WIDTH,
    GOAL_QUATERNION,
};

#[test]
fn test_adcs_state_machine_modes_masks_and_timeline() {
    let mut gt = GoalTable::new();
    let mut ct = ControllerTable::new();

    let mut detumble_act = BTreeMap::new();
    detumble_act.insert("Magnetorquer".to_string(), true);
    let detumble = ADCSMode::new(
        "detumble".to_string(),
        Some(Arc::new(NoGoal)),
        GOAL_NONE,
        [0.0; GOAL_PARAM_WIDTH],
        "NoGoal".to_string(),
        Some(Arc::new(BDot::new(5e4).unwrap())),
        BTreeMap::new(),
        detumble_act,
    )
    .unwrap();

    let mut point_sens = BTreeMap::new();
    point_sens.insert("Gyroscope".to_string(), false);
    let point = ADCSMode::new(
        "point".to_string(),
        Some(Arc::new(NadirGoal::new())),
        GOAL_NADIR,
        [0.0; GOAL_PARAM_WIDTH],
        "NadirGoal".to_string(),
        None,
        point_sens,
        BTreeMap::new(),
    )
    .unwrap();

    let q_goal = Arc::new(QuaternionGoal::new([1.0, 0.0, 0.0, 0.0], None).unwrap());
    let sm = ADCSStateMachine::new(
        "detumble".to_string(),
        vec![detumble.clone(), point],
        vec![
            (1.5, ADCSTimelineTarget::ModeName("point".to_string())),
            (
                3.0,
                ADCSTimelineTarget::GoalSpec {
                    goal: q_goal,
                    kind: GOAL_QUATERNION,
                    params: [1.0, 0.0, 0.0, 0.0],
                    label: "QuaternionGoal".to_string(),
                },
            ),
        ],
    )
    .unwrap();

    let rt = prepare_adcs_sm(
        Some(&sm),
        Some(&mut gt),
        Some(&mut ct),
        &["Gyroscope"],
        &[],
        &["Magnetorquer"],
    )
    .unwrap();
    assert!(rt.enabled);
    assert_eq!(rt.initial_mode, 0);
    assert_eq!(rt.mode_sensor_mask[1][0], 0);
    assert_eq!(rt.mode_mtq_mask[0][0], 1);

    let mut work = init_adcs_work(&rt);
    assert_eq!(work, [0, 0, 0]);

    let (m1, g1) = step_adcs_sm(0.5, &rt, &mut work, 1, -1);
    assert_eq!(m1, 1);
    assert_eq!(rt.goal_names[g1], "NadirGoal");

    let (m2, g2) = step_adcs_sm(3.0, &rt, &mut work, -1, -1);
    assert_eq!(m2, 1);
    assert_eq!(rt.goal_names[g2], "QuaternionGoal");

    let mut state = [0.0; 13];
    state[0] = 1.0;
    state[7] = 6_878_137.0;
    state[11] = 7_612.6;
    let (q_ref, w_ref) = evaluate_goal_default_mu(g2, &rt, &state, 2_461_000.5);
    assert!((q_ref[0] - 1.0).abs() < 1e-12);
    assert!(w_ref.iter().all(|&v| v.abs() < 1e-12));

    assert!(ADCSStateMachine::new(
        "detumble".to_string(),
        vec![detumble],
        vec![(1.0, ADCSTimelineTarget::ModeName("missing".to_string()))],
    )
    .is_err());
}
