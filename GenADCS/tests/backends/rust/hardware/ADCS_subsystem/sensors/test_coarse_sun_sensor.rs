use super::NativeCoarseSunSensor as CoarseSunSensor;
use std::f64::consts::PI;

fn q(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let n = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    [
        (angle / 2.0).cos(),
        axis[0] / n * (angle / 2.0).sin(),
        axis[1] / n * (angle / 2.0).sin(),
        axis[2] / n * (angle / 2.0).sin(),
    ]
}
fn sensor(
    normal: [f64; 3],
    sensitivity: f64,
    limit: f64,
    variance: f64,
    bias: f64,
    fov: f64,
    orientation: [f64; 4],
    misalignment: [f64; 4],
) -> CoarseSunSensor {
    CoarseSunSensor::new(
        [0.; 3],
        orientation,
        misalignment,
        normal,
        sensitivity,
        limit,
        variance,
        bias,
        fov,
    )
    .unwrap()
}
fn id() -> [f64; 4] {
    [1., 0., 0., 0.]
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}")
}

#[test]
fn measurement_uses_sensor_frame_cosine_projection() {
    let o = q([0., 0., 1.], PI / 2.);
    let m = q([1., 0., 0.], 0.2);
    let s = sensor([1., 0., 0.], 2., 10., 0.01, 0., PI / 3., o, m);
    close(s.noiseless_biasfree_measurement([0., 1., 0.], id()), 0.);
}

#[test]
fn backside_bias_clamps_and_saturation_applies() {
    let s = sensor([1., 0., 0.], 2., 10., 0.01, -0.5, PI / 3., id(), id());
    close(s.noiseless_biasfree_measurement([-1., 0., 0.], id()), 0.);
    close(s.noiseless_biased_measurement([-1., 0., 0.], id()), 0.);
    let s2 = sensor([1., 0., 0.], 3., 2., 0.01, 0.25, PI / 3., id(), id());
    close(s2.noiseless_biasfree_measurement([1., 0., 0.], id()), 2.);
    assert_eq!(s2.jacobian_bias(), 1.);
}

#[test]
fn noisy_measurements_are_finite_and_clamped() {
    let s = sensor([1., 0., 0.], 2., 3., 1e-8, 0., PI / 3., id(), id());
    for x in [
        s.noisy_biasfree_measurement([0.5, 0., (0.75_f64).sqrt()], id()),
        s.noisy_biased_measurement([0.5, 0., (0.75_f64).sqrt()], id()),
    ] {
        assert!(x.is_finite() && x >= 0. && x <= s.saturation_value);
    }
}

#[test]
fn attitude_jacobians_have_zero_error_scaling() {
    let s = sensor([0., 1., 0.], 2., 10., 0.01, 0., PI / 3., id(), id());
    let sun = [0., 0., 1.];
    let jq = s.jacobian_relative_quaternion(sun, id(), id());
    let jm = s.jacobian_relative_mrp(sun, id(), [0.; 3]);
    let ja = s.jacobian_relative_angle_vector(sun, id(), [0.; 3]);
    for i in 0..3 {
        close(jm[i], 2. * jq[i]);
        close(ja[i], 0.5 * jq[i]);
    }
}

#[test]
fn nonzero_jacobians_are_finite() {
    let s = sensor([1., 0., 0.], 2., 10., 0.01, 0., PI / 3., id(), id());
    let sun = [0.2, -0.1, 0.97];
    for j in [
        s.jacobian_relative_quaternion(sun, id(), q([1., 0., 0.], 0.1)),
        s.jacobian_relative_mrp(sun, id(), [0.02, -0.03, 0.01]),
        s.jacobian_relative_angle_vector(sun, id(), [0.02, -0.03, 0.01]),
    ] {
        assert!(j.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn availability_checks_visibility_front_side_and_field_of_view() {
    let s = sensor([0., 0., 1.], 2., 10., 0.01, 0., 0.3, id(), id());
    assert!(s.availability([0., 0., 1.], id(), true));
    assert!(!s.availability([0., 0., 1.], id(), false));
    assert!(!s.availability([0., 0., -1.], id(), true));
    assert!(!s.availability([0.5_f64.sin(), 0., 0.5_f64.cos()], id(), true));
}

#[test]
fn fisher_information_is_rank_one_and_zero_when_unavailable() {
    let s = sensor([0., 0., 1.], 2., 10., 0.5, 0., PI / 3., id(), id());
    let info = s.fisher_information_matrix([0., 0.2_f64.sin(), 0.2_f64.cos()], id(), true);
    assert!(info.iter().flatten().all(|x| x.is_finite()));
    assert!(s
        .fisher_information_matrix([0., 0., 1.], id(), false)
        .iter()
        .flatten()
        .all(|x| *x == 0.));
}

#[test]
fn invalid_configuration_is_rejected() {
    assert!(CoarseSunSensor::new([0.; 3], id(), id(), [0.; 3], 1., 1., 1., 0., 0.5).is_err());
    assert!(CoarseSunSensor::new([0.; 3], id(), id(), [1., 0., 0.], 0., 1., 1., 0., 0.5).is_err());
    assert!(CoarseSunSensor::new([0.; 3], id(), id(), [1., 0., 0.], 1., 1., 1., 0., PI).is_err());
}
