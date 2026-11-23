use super::*;
fn vector(_py: Python<'_>, value: &PyAny) -> Vec<f64> {
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

#[test]
fn rigid_body_and_wheel_dynamics_match_euler_equations() {
    Python::with_gil(|py| {
        let omega = [0.2, -0.4, 0.7];
        let torque = [0.03, -0.01, 0.02];
        let inertia = vec![vec![4., 0.2, -0.1], vec![0.2, 5., 0.3], vec![-0.1, 0.3, 6.]];
        let l = [
            4. * omega[0] + 0.2 * omega[1] - 0.1 * omega[2],
            0.2 * omega[0] + 5. * omega[1] + 0.3 * omega[2],
            -0.1 * omega[0] + 0.3 * omega[1] + 6. * omega[2],
        ];
        let cross = [
            omega[1] * l[2] - omega[2] * l[1],
            omega[2] * l[0] - omega[0] * l[2],
            omega[0] * l[1] - omega[1] * l[0],
        ];
        let result = rigid_body_angular_acceleration(
            py,
            omega.to_vec().into_py(py).as_ref(py),
            torque.to_vec().into_py(py).as_ref(py),
            inertia.clone().into_py(py).as_ref(py),
        )
        .unwrap();
        let actual = vector(py, result.as_ref(py));
        // The reference solve uses the full coupled inertia matrix.
        // Check the defining linear equation without duplicating a matrix solver.
        let reconstructed = [
            4. * actual[0] + 0.2 * actual[1] - 0.1 * actual[2],
            0.2 * actual[0] + 5. * actual[1] + 0.3 * actual[2],
            -0.1 * actual[0] + 0.3 * actual[1] + 6. * actual[2],
        ];
        near(
            &[
                reconstructed[0] + cross[0],
                reconstructed[1] + cross[1],
                reconstructed[2] + cross[2],
            ],
            &torque,
            1e-13,
        );

        let sphere = vec![vec![3., 0., 0.], vec![0., 3., 0.], vec![0., 0., 3.]];
        let spherical = rigid_body_angular_acceleration(
            py,
            omega.to_vec().into_py(py).as_ref(py),
            vec![0.3, -0.6, 0.9].into_py(py).as_ref(py),
            sphere.into_py(py).as_ref(py),
        )
        .unwrap();
        near(&vector(py, spherical.as_ref(py)), &[0.1, -0.2, 0.3], 1e-14);

        let axes = vec![vec![1., 0., 0.], vec![0., 0.6, 0.8]];
        let h = [0.04, -0.03];
        let wt = [0.01, 0.02, -0.04];
        let wheel = wheel_coupled_angular_acceleration(
            py,
            omega.to_vec().into_py(py).as_ref(py),
            h.to_vec().into_py(py).as_ref(py),
            axes.clone().into_py(py).as_ref(py),
            wt.to_vec().into_py(py).as_ref(py),
            inertia.clone().into_py(py).as_ref(py),
        )
        .unwrap();
        let wheel = vector(py, wheel.as_ref(py));
        let zero_h = vec![0., 0.].into_py(py);
        let no_wheel = wheel_coupled_angular_acceleration(
            py,
            omega.to_vec().into_py(py).as_ref(py),
            zero_h.as_ref(py),
            axes.into_py(py).as_ref(py),
            wt.to_vec().into_py(py).as_ref(py),
            inertia.into_py(py).as_ref(py),
        )
        .unwrap();
        assert!(wheel
            .iter()
            .zip(vector(py, no_wheel.as_ref(py)))
            .any(|(a, b)| (a - b).abs() > 1e-8));

        let alpha = [0.3, -0.4, 0.2];
        let wheel_torque = [0.2, -0.1, 0.05];
        let axes3 = vec![
            vec![1., 0., 0.],
            vec![0., 1., 0.],
            vec![
                std::f64::consts::FRAC_1_SQRT_2,
                0.,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
        ];
        let derivative = wheel_momentum_derivative(
            py,
            wheel_torque.to_vec().into_py(py).as_ref(py),
            alpha.to_vec().into_py(py).as_ref(py),
            axes3.into_py(py).as_ref(py),
        )
        .unwrap();
        near(
            &vector(py, derivative.as_ref(py)),
            &[-0.1, 0.3, 0.05 - 0.5 * std::f64::consts::FRAC_1_SQRT_2],
            1e-14,
        );
        assert!(wheel_momentum_derivative(
            py,
            Vec::<f64>::new().into_py(py).as_ref(py),
            alpha.to_vec().into_py(py).as_ref(py),
            Vec::<Vec<f64>>::new().into_py(py).as_ref(py)
        )
        .unwrap()
        .as_ref(py)
        .call_method0("tolist")
        .unwrap()
        .extract::<Vec<f64>>()
        .unwrap()
        .is_empty());
    });
}
