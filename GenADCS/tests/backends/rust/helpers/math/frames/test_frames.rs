use super::*;

#[test]
fn quaternion_rotation_is_orthogonal_and_invertible() {
    Python::with_gil(|py| {
        let q = vec![2.0_f64.sqrt() / 2.0, 0.0, 0.0, 2.0_f64.sqrt() / 2.0].into_py(py);
        let m = inertial_to_body_matrix(py, q.as_ref(py)).unwrap();
        let m = matrix(py, m.as_ref(py), "rotation").unwrap();
        assert!((m * m.transpose() - M::identity()).norm() < 1e-14);
        let v = vec![1.0, 0.0, 0.0].into_py(py);
        let result = inertial_to_body(py, v.as_ref(py), q.as_ref(py)).unwrap();
        let values: Vec<f64> = result.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!(values[0].abs() < 1e-14 && (values[1] - 1.0).abs() < 1e-14);
        let result = body_to_inertial(py, result.as_ref(py), q.as_ref(py)).unwrap();
        let values: Vec<f64> = result.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-14);
    });
}

#[test]
fn covariance_transform_matches_formula_and_preserves_symmetry() {
    Python::with_gil(|py| {
        let q = vec![0.5, 0.5, 0.5, 0.5].into_py(py);
        let m = inertial_to_body_matrix(py, q.as_ref(py)).unwrap();
        let covariance = vec![vec![4., 1., 0.], vec![1., 2., 0.], vec![0., 0., 3.]].into_py(py);
        let actual = transform_covariance(py, covariance.as_ref(py), m.as_ref(py)).unwrap();
        let actual = matrix(py, actual.as_ref(py), "result").unwrap();
        let rotation = matrix(py, m.as_ref(py), "rotation").unwrap();
        let input = M::from_row_slice(&[4.,1.,0.,1.,2.,0.,0.,0.,3.]);
        assert!((actual - rotation * input * rotation.transpose()).norm() < 1e-14);
        assert!((actual - actual.transpose()).norm() < 1e-14);
    });
}

#[test]
fn invalid_shapes_and_nonfinite_values_are_rejected() {
    Python::with_gil(|py| {
        let vector = vec![1.0, 2.0].into_py(py);
        let identity = vec![vec![1.,0.,0.],vec![0.,1.,0.],vec![0.,0.,1.]].into_py(py);
        assert!(transform_vector(py, vector.as_ref(py), identity.as_ref(py)).is_err());
        let bad_matrix = vec![vec![1., 0.], vec![0., 1.]].into_py(py);
        assert!(matrix(py, bad_matrix.as_ref(py), "matrix").is_err());
        let bad_q = vec![1.0, f64::NAN, 0.0, 0.0].into_py(py);
        assert!(inertial_to_body_matrix(py, bad_q.as_ref(py)).is_err());
        let bad_shape_q = vec![1.0, 0.0, 0.0].into_py(py);
        assert!(inertial_to_body_matrix(py, bad_shape_q.as_ref(py)).is_err());
        let bad_covariance = vec![vec![1., 0.], vec![0., 1.]].into_py(py);
        assert!(transform_covariance(py, bad_covariance.as_ref(py), identity.as_ref(py)).is_err());
    });
}
