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

#[test]
fn zonal_force_matches_closed_form_j2_off_the_equator() {
    // Regression test: the latitude term once had the wrong sign, which only
    // shows away from the equator (z != 0).
    let (mu, re, j2, mass) = (3.986004418e14_f64, 6378137.0_f64, 1.08262668e-3_f64, 12.0_f64);
    for p in [
        [7_000_000.0_f64, 0.0, 1_000_000.0],
        [4_000_000.0, 3_000_000.0, 4_500_000.0],
        [-2_500_000.0, 1_500_000.0, -6_000_000.0],
    ] {
        let f = zonal_gravity_force_native(p, mass, 0.0, mu, re, &[j2, 0., 0., 0., 0.]).unwrap();
        let [x, y, z] = p;
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        let k = 1.5 * j2 * mu * re * re / r.powi(5);
        let zz = 5.0 * z * z / r2;
        let c = -mu / (r2 * r);
        let expected = [c * x + k * x * (zz - 1.0), c * y + k * y * (zz - 1.0), c * z + k * z * (zz - 3.0)];
        for i in 0..3 {
            assert!((f[i] / mass - expected[i]).abs() < 1e-12, "axis {i}");
        }
    }
}
