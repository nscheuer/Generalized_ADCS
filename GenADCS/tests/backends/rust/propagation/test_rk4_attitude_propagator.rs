use crate::attitude_propagators::rk4_attitude_propagator::attitude_rk4;
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
fn torque_free_rk4_attitude_step_preserves_orbit_and_rate() {
    let runtime = torque_free_runtime();
    let state = vec![
        1.0, 0.0, 0.0, 0.0, // q
        0.0, 0.0, 0.1,      // w
        7.0e6, 0.0, 0.0,    // r
        0.0, 7500.0, 0.0,   // v
    ];
    let result = attitude_rk4(&state, 2_451_545.0, 1.0, &runtime).unwrap();
    let angle = 0.05_f64;
    assert!((result[0] - angle.cos()).abs() < 1e-8);
    assert!((result[3] + angle.sin()).abs() < 1e-8);
    assert!((result[6] - 0.1).abs() < 1e-12);
    assert_eq!(&result[7..13], &state[7..13]);
}
