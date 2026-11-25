use super::AutonomousStarTracker;

fn q(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let n = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    [
        (angle / 2.).cos(),
        axis[0] / n * (angle / 2.).sin(),
        axis[1] / n * (angle / 2.).sin(),
        axis[2] / n * (angle / 2.).sin(),
    ]
}
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn tracker(
    noise: [[f64; 3]; 3],
    orientation: [f64; 4],
    misalignment: [f64; 4],
) -> AutonomousStarTracker {
    AutonomousStarTracker::new(
        &[0.; 3],
        &orientation,
        &misalignment,
        1.,
        &noise.map(|r| r.to_vec()),
        &[0.; 3],
        std::f64::consts::PI,
    )
    .unwrap()
}
fn id() -> [f64; 4] {
    [1., 0., 0., 0.]
}
#[test]
fn fov_excludes_directions_inside_forward_cone() {
    let t = tracker([[1e-8, 0., 0.], [0., 1e-8, 0.], [0., 0., 1e-8]], id(), id());
    assert!(t.availability(Some(id()), Some([1., 0., 0.]), None));
    assert!(!t.availability(Some(id()), Some([0., 0., 1.]), None));
    assert!(t.availability(None, Some([0., 0., 1.]), None));
}
#[test]
fn mounting_quaternion_order_and_noiseless_variant() {
    let o = q([0., 0., 1.], 0.3);
    let m = q([1., 0., 0.], -0.2);
    let a = q([0., 1., 0.], 0.5);
    let t = tracker([[1e-8, 0., 0.], [0., 1e-8, 0.], [0., 0., 1e-8]], o, m);
    let expected = mul(m, mul(o, a));
    let actual = t.noiseless_biasfree_measurement(a);
    for i in 0..4 {
        assert!((actual[i] - expected[i]).abs() < 1e-14)
    }
    assert_eq!(t.noiseless_biased_measurement(a), actual);
}
#[test]
fn noisy_measurement_is_finite_and_normalized() {
    let t = tracker(
        [[1e-12, 0., 0.], [0., 1e-12, 0.], [0., 0., 1e-12]],
        id(),
        id(),
    );
    let measured = t.noisy_biasfree_measurement(q([1., 2., -1.], 0.7));
    assert!(measured.iter().all(|x| x.is_finite()));
    assert!((measured.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.).abs() < 1e-14);
    assert_eq!(t.noisy_biased_measurement(id()).len(), 4);
}
#[test]
fn residual_is_sign_invariant_and_jacobians_match_at_zero() {
    let t = tracker([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], id(), id());
    assert!(t.residual(id(), id()).iter().all(|x| x.abs() < 1e-14));
    let neg = id().map(|x| -x);
    assert!(t.residual(neg, id()).iter().all(|x| x.abs() < 1e-14));
    assert_eq!(
        t.jacobian_relative_quaternion(),
        [[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]
    );
    assert_eq!(
        t.jacobian_relative_mrp([0.; 3]),
        [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]
    );
    assert_eq!(
        t.jacobian_relative_angle_vector([0.; 3]),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
}
#[test]
fn information_is_zero_when_disabled() {
    let mut t = tracker([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], id(), id());
    assert!(t.fisher_information_matrix()[0][0] > 0.);
    t.sensor.enabled = false;
    assert_eq!(t.fisher_information_matrix(), [[0.; 3]; 3]);
    assert!(!t.availability(None, None, None));
}
