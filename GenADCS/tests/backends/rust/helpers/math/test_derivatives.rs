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
fn rotated(py: Python<'_>, q: &[f64], v: &[f64]) -> Vec<f64> {
    let qo = q.to_vec().into_py(py);
    let vo = v.to_vec().into_py(py);
    vector(
        py,
        _rotate_vector(py, qo.as_ref(py), vo.as_ref(py))
            .unwrap()
            .as_ref(py),
    )
}

#[test]
fn quaternion_derivatives_and_rate_matrix_are_consistent() {
    Python::with_gil(|py| {
        let mut q = [0.72, -0.2, 0.31, 0.57];
        let qnorm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.iter_mut().for_each(|x| *x /= qnorm);
        let omega = [0.4, -0.7, 0.2];
        let q_obj = q.to_vec().into_py(py);
        let w_obj = omega.to_vec().into_py(py);
        let qdot = quaternion_derivative(py, q_obj.as_ref(py), w_obj.as_ref(py)).unwrap();
        let qdot = vector(py, qdot.as_ref(py));
        let rate = quaternion_rate_matrix(py, q_obj.as_ref(py)).unwrap();
        let rate = matrix(py, rate.as_ref(py));
        let expected: Vec<f64> = rate
            .iter()
            .map(|row| row.iter().zip(omega).map(|(a, b)| a * b).sum())
            .collect();
        near(&qdot, &expected, 1e-14);
        let qdot_obj = qdot.clone().into_py(py);
        let recovered =
            angular_velocity_from_quaternion_derivative(py, q_obj.as_ref(py), qdot_obj.as_ref(py))
                .unwrap();
        near(&vector(py, recovered.as_ref(py)), &omega, 1e-14);
        assert!(q.iter().zip(&qdot).map(|(a, b)| a * b).sum::<f64>().abs() < 1e-14);
        assert_eq!((rate.len(), rate[0].len()), (4, 3));

        let raw = [0.4, -1.2, 0.7, 2.0];
        let raw_obj = raw.to_vec().into_py(py);
        let jacobian_obj = quaternion_normalization_jacobian(py, raw_obj.as_ref(py)).unwrap();
        let jacobian = matrix(py, jacobian_obj.as_ref(py));
        let step = 1e-7;
        let mut finite_diff = vec![vec![0.0; 4]; 4];
        for column in 0..4 {
            let mut plus = raw;
            let mut minus = raw;
            plus[column] += step;
            minus[column] -= step;
            let p: f64 = plus.iter().map(|x| x * x).sum::<f64>().sqrt();
            let m: f64 = minus.iter().map(|x| x * x).sum::<f64>().sqrt();
            for row in 0..4 {
                finite_diff[row][column] = (plus[row] / p - minus[row] / m) / (2.0 * step);
            }
        }
        matrix_near(&jacobian, &finite_diff, 2e-8);
        let radial: Vec<f64> = jacobian
            .iter()
            .map(|row| row.iter().zip(raw).map(|(a, b)| a * b).sum())
            .collect();
        near(&radial, &[0.0; 4], 1e-13);
        let zero = vec![0.0; 4].into_py(py);
        assert!(quaternion_normalization_jacobian(py, zero.as_ref(py)).is_err());

        let rotation_jacobian_obj = rotated_vector_quaternion_jacobian(
            py,
            q_obj.as_ref(py),
            vec![1.3, -0.7, 2.1].into_py(py).as_ref(py),
        )
        .unwrap();
        let rotation_jacobian = matrix(py, rotation_jacobian_obj.as_ref(py));
        let input_vector = [1.3, -0.7, 2.1];
        let mut rotation_fd = vec![vec![0.0; 4]; 3];
        for column in 0..4 {
            let mut plus = q;
            let mut minus = q;
            plus[column] += step;
            minus[column] -= step;
            let p = rotated(py, &plus, &input_vector);
            let m = rotated(py, &minus, &input_vector);
            for row in 0..3 {
                rotation_fd[row][column] = (p[row] - m[row]) / (2.0 * step);
            }
        }
        matrix_near(&rotation_jacobian, &rotation_fd, 3e-7);

        near(
            &vector(
                py,
                quaternion_to_euler321(py, vec![1., 0., 0., 0.].into_py(py).as_ref(py))
                    .unwrap()
                    .as_ref(py),
            ),
            &[0., 0., 0.],
            1e-14,
        );
        let half = std::f64::consts::FRAC_1_SQRT_2;
        near(
            &vector(
                py,
                quaternion_to_euler321(py, vec![half, half, 0., 0.].into_py(py).as_ref(py))
                    .unwrap()
                    .as_ref(py),
            ),
            &[std::f64::consts::FRAC_PI_2, 0., 0.],
            1e-14,
        );
        near(
            &vector(
                py,
                quaternion_to_euler321(py, vec![half, 0., 0., half].into_py(py).as_ref(py))
                    .unwrap()
                    .as_ref(py),
            ),
            &[0., 0., std::f64::consts::FRAC_PI_2],
            1e-14,
        );
    });
}
