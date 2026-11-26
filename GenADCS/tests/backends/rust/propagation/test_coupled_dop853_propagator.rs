use crate::{derivative_vec, dop853_vec, integrate_dop853_native, Runtime};

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
fn coupled_dop853_attempt_and_short_integration_succeed() {
    let runtime = torque_free_runtime();
    let initial = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0,
    ];
    let deriv = derivative_vec(&initial, 2_451_545.0, &runtime).unwrap();
    let atol = vec![1e-12; 13];
    let (next, _f_next, err) =
        dop853_vec(&initial, &deriv, 2_451_545.0, 0.5, &runtime, 1e-9, &atol).unwrap();
    assert!(err.is_finite() && err >= 0.0);
    assert!((next[0] - 0.025_f64.cos()).abs() < 1e-10);

    let (times, states) = integrate_dop853_native(
        initial,
        2_451_545.0,
        0.5,
        1.0,
        &runtime,
        1e-9,
        atol,
        10.0,
        1e-6,
        false,
        500,
        1,
        || Ok::<(), ()>(()),
    )
    .unwrap();
    assert!((times.last().unwrap() - 1.0).abs() < 1e-12);
    assert!((states.last().unwrap()[0] - 0.05_f64.cos()).abs() < 1e-10);
    assert!(states.last().unwrap()[8] > 0.0);
}
