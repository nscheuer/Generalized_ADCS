use crate::attitude_propagators::reference_attitude_propagator::set_reference;
use crate::Runtime;

fn basic_runtime(mu: f64) -> Runtime {
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    Runtime {
        mu,
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
        mtq_axes: Vec::new(),
        mtq_max_dipole: Vec::new(),
        dipole_commands: Vec::new(),
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
fn nadir_and_along_track_modes_produce_unit_quaternion_and_finite_rates() {
    let radius = 7.0e6;
    let speed = 7500.0;
    let mu = speed * speed * radius;
    let runtime = basic_runtime(mu);

    let mut nadir = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, radius, 0.0, 0.0, 0.0, speed, 0.0,
    ];
    set_reference(&mut nadir, 2_451_545.0, 0, &runtime).unwrap();
    let q_norm: f64 = nadir[..4].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((q_norm - 1.0).abs() < 1e-12);
    assert!(nadir[4..7].iter().all(|v| v.is_finite()));

    let mut along = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, radius, 0.0, 0.0, 0.0, speed, 0.0,
    ];
    set_reference(&mut along, 2_451_545.0, 1, &runtime).unwrap();
    let along_norm: f64 = along[..4].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((along_norm - 1.0).abs() < 1e-12);
}

#[test]
fn rejects_invalid_mode_and_missing_sun_ephemeris() {
    let runtime = basic_runtime(3.986_004_418e14);
    let mut state = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0,
    ];
    assert!(set_reference(&mut state, 2_451_545.0, 2, &runtime).is_err());
    assert!(set_reference(&mut state, 2_451_545.0, 99, &runtime).is_err());
}
