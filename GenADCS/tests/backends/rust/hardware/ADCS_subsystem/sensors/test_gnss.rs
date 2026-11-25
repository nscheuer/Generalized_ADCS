use super::GNSS;
fn sensor(noise: [[f64; 6]; 6], bias: [f64; 6]) -> GNSS {
    GNSS::new(&noise.map(|r| r.to_vec()), &bias).unwrap()
}
fn identity() -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}
#[test]
fn measurement_orders_position_then_velocity() {
    let s = sensor(identity(), [0.; 6]);
    assert_eq!(
        s.noiseless_biasfree_measurement([1., 2., 3.], [-4., 5., 6.]),
        [1., 2., 3., -4., 5., 6.]
    );
}
#[test]
fn bias_adds_to_all_channels_and_bias_jacobian_is_identity() {
    let b = [1., 2., 3., 0.1, 0.2, 0.3];
    let s = sensor(identity(), b);
    let actual = s.noiseless_biased_measurement([0., 1., 2.], [3., 4., 5.]);
    assert_eq!(actual, [1., 3., 5., 3.1, 4.2, 5.3]);
    assert_eq!(s.jacobian_bias(), identity());
}
#[test]
fn noisy_measurements_are_finite_six_vectors() {
    let s = sensor(identity(), [0.; 6]);
    for result in [
        s.noisy_biasfree_measurement([0.; 3], [1.; 3]),
        s.noisy_biased_measurement([0.; 3], [1.; 3]),
    ] {
        assert!(result.iter().all(|x| x.is_finite()));
    }
}
#[test]
fn position_and_velocity_jacobians_have_expected_blocks() {
    let s = sensor(identity(), [0.; 6]);
    let jp = s.jacobian_position();
    let jv = s.jacobian_velocity();
    for i in 0..6 {
        for j in 0..3 {
            assert_eq!(jp[i][j], if i < 3 && i == j { 1. } else { 0. });
            assert_eq!(jv[i][j], if i >= 3 && i - 3 == j { 1. } else { 0. });
        }
    }
}
#[test]
fn attitude_jacobians_are_zero_and_availability_is_live() {
    let mut s = sensor(identity(), [0.; 6]);
    assert_eq!(s.jacobian_relative_quaternion(), [[0.; 3]; 6]);
    assert_eq!(s.jacobian_relative_mrp(), [[0.; 3]; 6]);
    assert_eq!(s.jacobian_relative_angle_vector(), [[0.; 3]; 6]);
    assert!(s.availability());
    s.enabled = false;
    assert!(!s.availability());
}
