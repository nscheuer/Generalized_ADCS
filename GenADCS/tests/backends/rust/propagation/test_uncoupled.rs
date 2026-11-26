use crate::separate_solver::integrate_uncoupled_native;
use crate::Runtime;

fn torque_free_runtime() -> Runtime {
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    Runtime {
        mu: 3.986_004_418e14,
        radius: 6_378_137.0,
        mass: 10.0,
        inertia: eye,
        inverse_inertia: eye,
        zonals: Vec::new(),
        normals: Vec::new(),
        arms: Vec::new(),
        areas: Vec::new(),
        drag_coefficients: Vec::new(),
        optical_coefficients: Vec::new(),
        dipole: [0.0; 3],
        drag_force: false,
        drag_torque: false,
        srp_force: false,
        srp_torque: false,
        gravity_gradient: false,
        third_body: false,
        ephemeris: None,
        magnetic: None,
        wheel_axes: Vec::new(),
        wheel_inertias: Vec::new(),
        inverse_bus_inertia: eye,
        wheel_commands: Vec::new(),
        wheel_max_momentum: Vec::new(),
    }
}

#[test]
fn uncoupled_rk4_and_reference_integrate_short_horizon() {
    let runtime = torque_free_runtime();
    let initial = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0,
    ];

    let (times_rk4, states_rk4) = integrate_uncoupled_native(
        initial.clone(),
        2_451_545.0,
        0.25,
        1.1,
        &runtime,
        -1,
        false,
        500,
        3,
        || true,
    )
    .unwrap();
    assert_eq!(times_rk4.len(), 3);
    assert!((times_rk4[2] - 1.1).abs() < 1e-12);
    assert!((states_rk4[2][0] - 0.055_f64.cos()).abs() < 1e-9);
    assert!((states_rk4[2][3] + 0.055_f64.sin()).abs() < 1e-9);
    assert!(states_rk4[2][8] > 0.0);

    let (times_ref, states_ref) = integrate_uncoupled_native(
        initial,
        2_451_545.0,
        0.25,
        0.5,
        &runtime,
        0,
        false,
        500,
        1,
        || true,
    )
    .unwrap();
    assert_eq!(times_ref.len(), 3);
    let q_norm: f64 = states_ref[0][..4].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((q_norm - 1.0).abs() < 1e-12);
    assert!(states_ref[2][8] > 0.0);
}
