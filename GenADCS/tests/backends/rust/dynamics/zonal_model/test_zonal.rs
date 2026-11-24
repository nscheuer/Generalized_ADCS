use super::*;
#[test]
fn zonal_force_physics_and_invalid_inputs() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let p = vec![7e6, 0., 0.].into_py(py);
        let j = vec![
            1.08262668e-3,
            -2.5324105e-6,
            -1.6198976e-6,
            -2.2729608e-7,
            5.406812e-7,
        ]
        .into_py(py);
        let f = zonal_gravity_force(
            py,
            p.as_ref(py),
            10.,
            0.,
            3.986004418e14,
            6378137.,
            j.as_ref(py),
        )
        .unwrap();
        let a: Vec<f64> = f
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!(a[0] < 0.);
        let doubled = zonal_gravity_force(
            py,
            p.as_ref(py),
            20.,
            0.,
            3.986004418e14,
            6378137.,
            j.as_ref(py),
        )
        .unwrap();
        let b: Vec<f64> = doubled
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!((b[0] - 2. * a[0]).abs() < 1e-9);
        let z = vec![0., 0., 0.].into_py(py);
        assert!(
            zonal_gravity_force(py, z.as_ref(py), 10., 0., 3.986e14, 6378137., j.as_ref(py))
                .is_err()
        );
        let bad = vec![1., 2.].into_py(py);
        assert!(zonal_gravity_force(
            py,
            p.as_ref(py),
            10.,
            0.,
            3.986e14,
            6378137.,
            bad.as_ref(py)
        )
        .is_ok());
    });
}
