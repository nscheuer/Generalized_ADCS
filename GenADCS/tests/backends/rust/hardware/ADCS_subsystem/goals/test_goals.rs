use ephemeris_native::NativeEphemeris;
use frames_native::inertial_to_body_matrix_native;
use math_native::utc_jd_to_tdb_jd_native;

use super::goal::{
    validate_quaternion, validate_reference_state, validate_vector, Goal, DEFAULT_MU,
};
use super::sun_goal::default_de421_path;
use super::{
    AlongTrackGoal, AngularRateGoal, CoordinateFMCGoal, CoordinateGoal, CrossTrackGoal, NadirGoal,
    NoGoal, QuaternionGoal, SunGoal, TwoVectorGoal, VectorGoal, VelocityGoal, ZenithGoal,
};

fn circular_equatorial_state() -> [f64; 13] {
    let r = 7_000_000.0;
    let v = (DEFAULT_MU / r).sqrt();
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, r, 0.0, 0.0, 0.0, v, 0.0]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm4(q: [f64; 4]) -> f64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn body_axis_in_inertial(q: [f64; 4], axis: [f64; 3]) -> [f64; 3] {
    let dcm_b_i = inertial_to_body_matrix_native(q);
    let dcm_i_b = dcm_b_i.transpose();
    let v = dcm_i_b * nalgebra::Vector3::new(axis[0], axis[1], axis[2]);
    [v[0], v[1], v[2]]
}

#[test]
fn test_validation_helpers() {
    let valid = circular_equatorial_state();
    assert!(validate_reference_state(&valid).is_ok());
    assert!(validate_reference_state(&valid[..12]).is_err());

    let mut bad_nan = valid;
    bad_nan[7] = f64::NAN;
    assert!(validate_reference_state(&bad_nan).is_err());

    let mut zero_q = valid;
    zero_q[0] = 0.0;
    assert!(validate_reference_state(&zero_q).is_err());

    let q = validate_quaternion([2.0, 0.0, 0.0, 0.0]).unwrap();
    assert!((q[0] - 1.0).abs() < 1e-12);
    assert!(validate_quaternion([0.0, 0.0, 0.0, 0.0]).is_err());
    assert!(validate_quaternion([1.0, f64::INFINITY, 0.0, 0.0]).is_err());

    let v = validate_vector([0.0, 3.0, 4.0], "v").unwrap();
    assert!((norm3(v) - 1.0).abs() < 1e-12);
    assert!(validate_vector([0.0, 0.0, 0.0], "v").is_err());
    assert!(validate_vector([1.0, f64::NAN, 0.0], "v").is_err());
}

#[test]
fn test_no_goal() {
    let goal = NoGoal::new();
    let (q, r) = goal
        .reference(&circular_equatorial_state(), 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(q.is_none());
    assert!(r.is_none());
    assert!(goal.reference(&[0.0; 10], 2451545.0, DEFAULT_MU).is_err());
}

#[test]
fn test_quaternion_goal() {
    let goal = QuaternionGoal::new([2.0, 0.0, 0.0, 0.0], None).unwrap();
    let (q, r) = goal
        .reference(&circular_equatorial_state(), 2451545.0, DEFAULT_MU)
        .unwrap();
    assert_eq!(q.unwrap(), [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.unwrap(), [0.0, 0.0, 0.0]);

    let custom_rate = [0.01, -0.02, 0.03];
    let goal2 = QuaternionGoal::new([0.0, 1.0, 1.0, 0.0], Some(custom_rate)).unwrap();
    let (q2, r2) = goal2
        .reference(&circular_equatorial_state(), 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!((norm4(q2.unwrap()) - 1.0).abs() < 1e-12);
    assert_eq!(r2.unwrap(), custom_rate);

    assert!(QuaternionGoal::new([0.0; 4], None).is_err());
    assert!(QuaternionGoal::new([1.0, 0.0, 0.0, 0.0], Some([f64::NAN, 0.0, 0.0])).is_err());
}

#[test]
fn test_angular_rate_goal() {
    let rate = [0.1, -0.2, 0.3];
    let goal = AngularRateGoal::new(rate).unwrap();
    let (q, r) = goal
        .reference(&circular_equatorial_state(), 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(q.is_none());
    assert_eq!(r.unwrap(), rate);

    assert!(AngularRateGoal::new([0.0, f64::INFINITY, 0.0]).is_err());
}

#[test]
fn test_vector_and_two_vector_goals() {
    let state = circular_equatorial_state();
    // Align body +Z to inertial +X, with +X roll to +Y -> body +Y = +Z
    let vg = VectorGoal::new(
        [1.0, 0.0, 0.0],
        Some([0.0, 0.0, 1.0]),
        Some([0.0, 1.0, 0.0]),
        Some([1.0, 0.0, 0.0]),
    )
    .unwrap();
    let (q, r) = vg.reference(&state, 2451545.0, DEFAULT_MU).unwrap();
    let q = q.unwrap();
    assert!((norm4(q) - 1.0).abs() < 1e-12);
    assert_eq!(r.unwrap(), [0.0, 0.0, 0.0]);
    let z_inert = body_axis_in_inertial(q, [0.0, 0.0, 1.0]);
    assert!(dot3(z_inert, [1.0, 0.0, 0.0]) > 1.0 - 1e-12);
    let x_inert = body_axis_in_inertial(q, [1.0, 0.0, 0.0]);
    assert!(dot3(x_inert, [0.0, 1.0, 0.0]) > 1.0 - 1e-12);

    // Parallel roll rejected
    assert!(VectorGoal::new([1.0, 0.0, 0.0], None, None, None).is_err());
    assert!(VectorGoal::new(
        [0.0, 1.0, 0.0],
        Some([1.0, 0.0, 0.0]),
        Some([1.0, 0.0, 0.0]),
        Some([1.0, 0.0, 0.0]),
    )
    .is_err());

    let tvg = TwoVectorGoal::new(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    .unwrap();
    let (tq, tr) = tvg.reference(&state, 2451545.0, DEFAULT_MU).unwrap();
    let tq = tq.unwrap();
    assert_eq!(tr.unwrap(), [0.0, 0.0, 0.0]);
    assert!(dot3(body_axis_in_inertial(tq, [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]) > 1.0 - 1e-12);
    assert!(dot3(body_axis_in_inertial(tq, [1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]) > 1.0 - 1e-12);

    assert!(TwoVectorGoal::new(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .is_err());
    assert!(TwoVectorGoal::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, -3.0]],
    )
    .is_err());
}

#[test]
fn test_orbit_goals() {
    let state = circular_equatorial_state();
    let r_mag = 7_000_000.0_f64;
    let mean_motion = (DEFAULT_MU / r_mag.powi(3)).sqrt();

    let (q, rate) = NadirGoal::new()
        .reference(&state, 2451545.0, DEFAULT_MU)
        .unwrap();
    let q = q.unwrap();
    let rate = rate.unwrap();
    assert!((norm4(q) - 1.0).abs() < 1e-12);
    assert!(dot3(body_axis_in_inertial(q, [0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0]) > 1.0 - 1e-12);
    assert!((norm3(rate) - mean_motion).abs() < 1e-12);

    let (qz, rz) = ZenithGoal::new()
        .reference(&state, 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(dot3(body_axis_in_inertial(qz.unwrap(), [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]) > 1.0 - 1e-12);
    assert!((norm3(rz.unwrap()) - mean_motion).abs() < 1e-12);

    let (qa, ra) = AlongTrackGoal::new()
        .reference(&state, 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(dot3(body_axis_in_inertial(qa.unwrap(), [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]) > 1.0 - 1e-12);
    assert!((norm3(ra.unwrap()) - mean_motion).abs() < 1e-12);

    let (qc, rc) = CrossTrackGoal::new()
        .reference(&state, 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(dot3(body_axis_in_inertial(qc.unwrap(), [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]) > 1.0 - 1e-12);
    assert!((norm3(rc.unwrap()) - mean_motion).abs() < 1e-12);

    let (qv, rv) = VelocityGoal::new()
        .reference(&state, 2451545.0, DEFAULT_MU)
        .unwrap();
    assert!(dot3(body_axis_in_inertial(qv.unwrap(), [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]) > 1.0 - 1e-12);
    assert!((norm3(rv.unwrap()) - mean_motion).abs() < 1e-12);

    // Invalid epoch_jd and zero position
    assert!(NadirGoal::new()
        .reference(&state, f64::NAN, DEFAULT_MU)
        .is_err());
    let mut zero_pos = state;
    zero_pos[7] = 0.0;
    assert!(NadirGoal::new()
        .reference(&zero_pos, 2451545.0, DEFAULT_MU)
        .is_err());
}

#[test]
fn test_sun_goal() {
    let state = circular_equatorial_state();
    let epoch_jd = 2451545.0;
    let sun_goal = SunGoal::new().unwrap();
    let (q, rate) = sun_goal.reference(&state, epoch_jd, DEFAULT_MU).unwrap();
    let q = q.unwrap();
    let rate = rate.unwrap();
    assert!((norm4(q) - 1.0).abs() < 1e-12);
    assert!(rate.iter().all(|v| v.is_finite()));

    // Verify body +Z aligns with Sun vector from DE421
    let eph = NativeEphemeris::from_path(&default_de421_path()).unwrap();
    let tdb_jd = utc_jd_to_tdb_jd_native(epoch_jd).unwrap();
    let sun_km = eph.position_km(0, tdb_jd).unwrap();
    let sun_rel = [
        sun_km[0] * 1000.0 - state[7],
        sun_km[1] * 1000.0 - state[8],
        sun_km[2] * 1000.0 - state[9],
    ];
    let n = norm3(sun_rel);
    let sun_dir = [sun_rel[0] / n, sun_rel[1] / n, sun_rel[2] / n];
    let z_inert = body_axis_in_inertial(q, [0.0, 0.0, 1.0]);
    assert!(dot3(z_inert, sun_dir) > 1.0 - 1e-12);
}

#[test]
fn test_coordinate_goal() {
    assert!(CoordinateGoal::new(-91.0, 0.0, None).is_err());
    assert!(CoordinateGoal::new(91.0, 0.0, None).is_err());
    assert!(CoordinateGoal::new(0.0, -181.0, None).is_err());
    assert!(CoordinateGoal::new(0.0, 181.0, None).is_err());
    assert!(CoordinateGoal::new(0.0, 0.0, Some(-7000.0)).is_err());
    assert!(CoordinateGoal::new(f64::NAN, 0.0, None).is_err());

    let eq = CoordinateGoal::new(0.0, 0.0, Some(0.0)).unwrap();
    assert!((eq.target_ecef_m[0] - 6378137.0).abs() < 1e-6);
    assert!(eq.target_ecef_m[1].abs() < 1e-9);
    assert!(eq.target_ecef_m[2].abs() < 1e-9);

    let state = circular_equatorial_state();
    let (q, rate) = eq.reference(&state, 2451545.0, DEFAULT_MU).unwrap();
    let q = q.unwrap();
    let rate = rate.unwrap();
    assert!((norm4(q) - 1.0).abs() < 1e-12);
    assert!(rate.iter().all(|v| v.is_finite()));
    assert!(norm3(rate) > 0.0);

    assert!(eq.reference(&state, f64::INFINITY, DEFAULT_MU).is_err());
}

#[test]
fn test_coordinate_fmc_goal() {
    assert!(CoordinateFMCGoal::new(0.0, 0.0, None, Some(0)).is_err());
    assert!(CoordinateFMCGoal::new(0.0, 0.0, None, Some(11)).is_err());
    assert!(CoordinateFMCGoal::new(0.0, 0.0, None, Some(-1)).is_err());

    let state = circular_equatorial_state();
    // FMC level 1 -> tilt = acos(1) = 0 -> body +Z points straight nadir (-X)
    let fmc1 = CoordinateFMCGoal::new(10.0, 20.0, Some(0.0), Some(1)).unwrap();
    let (q1, r1) = fmc1.reference(&state, 2451545.0, DEFAULT_MU).unwrap();
    let q1 = q1.unwrap();
    assert!((norm4(q1) - 1.0).abs() < 1e-12);
    assert!(r1.unwrap().iter().all(|v| v.is_finite()));
    let z1 = body_axis_in_inertial(q1, [0.0, 0.0, 1.0]);
    assert!(dot3(z1, [-1.0, 0.0, 0.0]) > 1.0 - 1e-12);

    // FMC level 2 -> tilt = acos(0.5) = 60 deg from nadir -> dot(z, nadir) == 0.5
    let fmc2 = CoordinateFMCGoal::new(10.0, 20.0, Some(0.0), Some(2)).unwrap();
    let (q2, r2) = fmc2.reference(&state, 2451545.0, DEFAULT_MU).unwrap();
    let q2 = q2.unwrap();
    assert!((norm4(q2) - 1.0).abs() < 1e-12);
    assert!(r2.unwrap().iter().all(|v| v.is_finite()));
    let z2 = body_axis_in_inertial(q2, [0.0, 0.0, 1.0]);
    assert!((dot3(z2, [-1.0, 0.0, 0.0]) - 0.5).abs() < 1e-12);
}
