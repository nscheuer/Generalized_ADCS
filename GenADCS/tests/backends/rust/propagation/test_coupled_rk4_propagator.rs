use crate::{integrate_rk4_native, rk4_vec, Runtime};

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
        mtq_axes: Vec::new(),
        mtq_max_dipole: Vec::new(),
        dipole_commands: Vec::new(),
    }
}

#[test]
fn coupled_rk4_step_and_short_integration_advance_attitude_and_orbit() {
    let runtime = torque_free_runtime();
    let initial = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0,
    ];
    let stepped = rk4_vec(&initial, 2_451_545.0, 1.0, &runtime).unwrap();
    assert!((stepped[0] - 0.05_f64.cos()).abs() < 1e-8);
    assert!((stepped[3] + 0.05_f64.sin()).abs() < 1e-8);
    assert!(stepped[8] > 0.0);

    let (times, states) = integrate_rk4_native(
        initial,
        2_451_545.0,
        0.25,
        1.1,
        &runtime,
        false,
        500,
        3,
        || Ok::<(), ()>(()),
    )
    .unwrap();
    assert_eq!(times.len(), 3);
    assert!((times[2] - 1.1).abs() < 1e-12);
    assert!((states[2][0] - 0.055_f64.cos()).abs() < 1e-9);
    assert!((states[2][3] + 0.055_f64.sin()).abs() < 1e-9);
    assert!(states[2][8] > 0.0);
}
