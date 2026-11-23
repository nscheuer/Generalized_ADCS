use super::*;
fn vector(_py: Python<'_>, value: &PyAny) -> Vec<f64> {
    value.call_method0("tolist").unwrap().extract().unwrap()
}
fn matrix(_py: Python<'_>, value: &PyAny) -> Vec<Vec<f64>> {
    value.call_method0("tolist").unwrap().extract().unwrap()
}
fn near(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= tolerance,
            "{actual:?} != {expected:?} (tol {tolerance})"
        );
    }
}
fn matrix_near(actual: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        near(a, e, tolerance);
    }
}

#[test]
fn vector_helpers_obey_cross_norm_and_skew_identities() {
    Python::with_gil(|py| {
        let a = vec![1., 2., 3.].into_py(py);
        let b = vec![-4., 5., 2.].into_py(py);
        let ab = cross(py, a.as_ref(py), b.as_ref(py)).unwrap();
        near(&vector(py, ab.as_ref(py)), &[-11., -14., 13.], 1e-14);
        let ba = cross(py, b.as_ref(py), a.as_ref(py)).unwrap();
        let ba = vector(py, ba.as_ref(py));
        near(&ba, &[11., 14., -13.], 1e-14);

        let x = vec![3., 4., 0.].into_py(py);
        assert!((norm(py, x.as_ref(py)).unwrap() - 5.).abs() < 1e-14);
        near(
            &vector(py, normalize(py, x.as_ref(py)).unwrap().as_ref(py)),
            &[0.6, 0.8, 0.],
            1e-14,
        );
        near(
            &vector(py, unit(py, x.as_ref(py)).unwrap().as_ref(py)),
            &[0.6, 0.8, 0.],
            1e-14,
        );
        let zero = vec![0., 0., 0.].into_py(py);
        near(
            &vector(py, normalize(py, zero.as_ref(py)).unwrap().as_ref(py)),
            &[0.; 3],
            0.,
        );
        assert!(unit(py, zero.as_ref(py)).is_err());

        let v = vec![1.5, -2., 4.].into_py(py);
        let m = skew(py, v.as_ref(py)).unwrap();
        let m = matrix(py, m.as_ref(py));
        matrix_near(
            &m,
            &[vec![0., -4., -2.], vec![4., 0., -1.5], vec![2., 1.5, 0.]],
            1e-14,
        );
        let product: Vec<f64> = m
            .iter()
            .map(|row| row.iter().zip([-4., 5., 2.]).map(|(x, y)| x * y).sum())
            .collect();
        near(&product, &[-24., -19., -0.5], 1e-14);
    });
}
