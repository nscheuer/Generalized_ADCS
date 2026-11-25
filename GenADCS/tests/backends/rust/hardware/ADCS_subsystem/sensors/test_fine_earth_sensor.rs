use super::NativeFineEarthSensor as FineEarthSensor;

fn id() -> [f64; 4] {
    [1., 0., 0., 0.]
}
fn q(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let n = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    [
        (angle / 2.).cos(),
        axis[0] / n * (angle / 2.).sin(),
        axis[1] / n * (angle / 2.).sin(),
        axis[2] / n * (angle / 2.).sin(),
    ]
}
fn make(
    noise: [[f64; 2]; 2],
    bias: [f64; 2],
    fov: [f64; 2],
    orientation: [f64; 4],
    misalignment: [f64; 4],
) -> FineEarthSensor {
    FineEarthSensor::new([0.; 3], orientation, misalignment, noise, bias, fov).unwrap()
}
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        (1. - 2. * (y * y + z * z)) * v[0]
            + 2. * (x * y - w * z) * v[1]
            + 2. * (x * z + w * y) * v[2],
        2. * (x * y + w * z) * v[0]
            + (1. - 2. * (x * x + z * z)) * v[1]
            + 2. * (y * z - w * x) * v[2],
        2. * (x * z - w * y) * v[0]
            + 2. * (y * z + w * x) * v[1]
            + (1. - 2. * (x * x + y * y)) * v[2],
    ]
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}")
}

#[test]
fn angles_are_measured_in_mounted_sensor_frame() {
    let o = q([0., 1., 0.], 0.2);
    let m = q([1., 0., 0.], -0.1);
    let att = q([0., 0., 1.], 0.3);
    let earth = [0.2, -0.1, 0.97];
    let s = make([[1e-6, 0.], [0., 2e-6]], [0.; 2], [0.6, 0.7], o, m);
    let v = rotate(mul(mul(m, o), att), earth);
    let actual = s.noiseless_biasfree_measurement(earth, att);
    close(actual[0], v[0].atan2(v[2]));
    close(actual[1], v[1].atan2(v[2]));
}

#[test]
fn bias_noise_and_bias_jacobian() {
    let s = make(
        [[1e-8, 0.], [0., 2e-8]],
        [-0.01, 0.02],
        [0.6, 0.7],
        id(),
        id(),
    );
    let earth = [0.1, 0.2, 0.97];
    let base = s.noiseless_biasfree_measurement(earth, id());
    let biased = s.noiseless_biased_measurement(earth, id());
    close(biased[0], base[0] - 0.01);
    close(biased[1], base[1] + 0.02);
    assert_eq!(s.jacobian_bias(), [[1., 0.], [0., 1.]]);
    for v in [
        s.noisy_biasfree_measurement(earth, id()),
        s.noisy_biased_measurement(earth, id()),
    ] {
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn attitude_jacobians_have_zero_error_scaling_and_correct_shapes() {
    let s = make([[1e-6, 0.], [0., 2e-6]], [0.; 2], [0.6, 0.7], id(), id());
    let earth = [0.1, 0.2, 0.97];
    let jq = s.jacobian_relative_quaternion(earth, id(), id());
    let jm = s.jacobian_relative_mrp(earth, id(), [0.; 3]);
    let ja = s.jacobian_relative_angle_vector(earth, id(), [0.; 3]);
    for i in 0..2 {
        for j in 0..3 {
            close(jm[i][j], 2. * jq[i][j]);
            close(ja[i][j], 0.5 * jq[i][j]);
        }
    }
    let nonzero = s.jacobian_relative_quaternion(earth, id(), q([1., 2., 0.], 0.1));
    assert!(nonzero.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn availability_checks_visibility_and_both_fov_axes() {
    let s = make([[1., 0.], [0., 1.]], [0.; 2], [0.2, 0.3], id(), id());
    assert!(s.availability([0., 0., 1.], id(), true));
    assert!(!s.availability([0., 0., 1.], id(), false));
    assert!(!s.availability([0.25_f64.sin(), 0., 0.25_f64.cos()], id(), true));
    assert!(!s.availability([0., 0.4_f64.sin(), 0.4_f64.cos()], id(), true));
}

#[test]
fn fisher_information_is_finite_and_zero_when_unavailable() {
    let s = make(
        [[0.01, 0.], [0., 0.02]],
        [0.; 2],
        [0.6, 0.7],
        id(),
        q([1., 0., 0.], 0.1),
    );
    let info = s.fisher_information_matrix([0.1, 0.2, 0.97], id(), true);
    assert!(info.iter().flatten().all(|x| x.is_finite()));
    assert!(s
        .fisher_information_matrix([0.1, 0.2, 0.97], id(), false)
        .iter()
        .flatten()
        .all(|x| *x == 0.));
}

#[test]
fn covariance_must_be_positive_definite() {
    assert!(
        FineEarthSensor::new([0.; 3], id(), id(), [[1., 0.], [0., -1.]], [0.; 2], [1.; 2]).is_err()
    );
}
