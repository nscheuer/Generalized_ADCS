use super::*;
#[test]
fn residual_dipole_cross_product_and_malformed_inputs() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let m = vec![1., 0., 0.].into_py(py);
        let b = vec![0., 0., 2e-5].into_py(py);
        let q = vec![1., 0., 0., 0.].into_py(py);
        let t =
            static_residual_dipole_torque(py, m.as_ref(py), b.as_ref(py), q.as_ref(py)).unwrap();
        let a: Vec<f64> = t
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!((a[1] + 2e-5).abs() < 1e-14);
        let bad = vec![1., 2.].into_py(py);
        assert!(
            static_residual_dipole_torque(py, bad.as_ref(py), b.as_ref(py), q.as_ref(py)).is_err()
        );
    });
}
