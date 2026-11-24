use super::*;
#[test]
fn canonicalization_shadow_and_kernel_roundtrip() {
    Python::with_gil(|py| {
        let raw = vec![2., 0., 0.].into_py(py);
        let p = MRP::new(py, Some(raw.as_ref(py))).unwrap();
        assert!((p.p[0] + 0.5).abs() < 1e-12);
        assert!(p.norm() <= 1.);
        let original = vec![0.2, 0.3, 0.4].into_py(py);
        let shadow = mrp_shadow_set(py, original.as_ref(py)).unwrap();
        let restored = mrp_shadow_set(py, shadow.as_ref(py)).unwrap();
        let r = fixed::<3>(py, restored.as_ref(py), "p").unwrap();
        for i in 0..3 {
            assert!((r[i] - [0.2, 0.3, 0.4][i]).abs() < 1e-12);
        }
        let q = mrp_to_quaternion(py, original.as_ref(py)).unwrap();
        let back = quaternion_to_mrp(py, q.as_ref(py)).unwrap();
        let b = fixed::<3>(py, back.as_ref(py), "p").unwrap();
        for i in 0..3 {
            assert!((b[i] - [0.2, 0.3, 0.4][i]).abs() < 1e-12);
        }
        let zero = vec![0., 0., 0.].into_py(py);
        assert!(mrp_shadow_set(py, zero.as_ref(py)).is_err());
        assert!(MRP::identity().switch_shadow_set().is_err());
    });
}
#[test]
fn class_constructors_rotation_and_group_operations() {
    Python::with_gil(|py| {
        let axis = vec![0., 0., 1.].into_py(py);
        let p = MRP::from_axis_angle(py, axis.as_ref(py), std::f64::consts::FRAC_PI_2).unwrap();
        let x = vec![1., 0., 0.].into_py(py);
        let y = p.rotate(py, x.as_ref(py)).unwrap();
        assert!((fixed::<3>(py, y.as_ref(py), "y").unwrap()[1] - 1.).abs() < 1e-12);
        let back = p.body_to_inertial(py, y.as_ref(py)).unwrap();
        assert!((fixed::<3>(py, back.as_ref(py), "x").unwrap()[0] - 1.).abs() < 1e-12);
        let q = p.to_quaternion(py).unwrap();
        let m = p.to_dcm(py).unwrap();
        let pq = MRP::from_quaternion(py, q.into_py(py).as_ref(py)).unwrap();
        assert!((pq.p[2] - p.p[2]).abs() < 1e-12);
        let pd = MRP::from_dcm(py, pymat(py, &m.m).as_ref(py)).unwrap();
        assert!((pd.p[2] - p.p[2]).abs() < 1e-12);
        let inverse = p.inverse(py).unwrap();
        let product = p.multiply(py, inverse.into_py(py).as_ref(py)).unwrap();
        assert!(product.norm() < 1e-12);
        let shadow = p.switch_shadow_set().unwrap();
        assert!((shadow.to_dcm(py).unwrap().m - m.m).norm() < 1e-12);
    });
}
