use super::*;

#[test]
fn frame_matrices_are_orthogonal_and_inverse_pairs() {
    Python::with_gil(|py| {
        let jd = 2451545.0;
        let a = gcrf_to_ecef_matrix(py, jd, jd, 1e-6, -2e-6).unwrap();
        let b = ecef_to_gcrf_matrix(py, jd, jd, 1e-6, -2e-6).unwrap();
        let a = matrix(py, a.as_ref(py), "gcrf_to_ecef").unwrap();
        let b = matrix(py, b.as_ref(py), "ecef_to_gcrf").unwrap();
        assert!((a * a.transpose() - M::identity()).norm() < 2e-12);
        assert!((b - a.transpose()).norm() < 1e-14);
        let a = gcrf_to_teme_matrix(py, jd).unwrap();
        let b = teme_to_gcrf_matrix(py, jd).unwrap();
        assert!((matrix(py, b.as_ref(py), "b").unwrap() - matrix(py, a.as_ref(py), "a").unwrap().transpose()).norm() < 1e-14);
    });
}

#[test]
fn lvlh_axes_and_degenerate_orbits() {
    Python::with_gil(|py| {
        let p = vec![7e6, 0., 0.].into_py(py);
        let v = vec![0., 7500., 0.].into_py(py);
        let result = gcrf_to_lvlh_matrix(py, p.as_ref(py), v.as_ref(py)).unwrap();
        assert!((matrix(py, result.as_ref(py), "lvlh").unwrap() - M::identity()).norm() < 1e-14);
        let zero = vec![0.; 3].into_py(py);
        assert!(gcrf_to_lvlh_matrix(py, zero.as_ref(py), v.as_ref(py)).is_err());
        assert!(gcrf_to_lvlh_matrix(py, p.as_ref(py), p.as_ref(py)).is_err());
    });
}

#[test]
fn context_errors_vector_roundtrip_and_frame_parse() {
    Python::with_gil(|py| {
        assert!(frame_rotation(py, "GCRF", "ECEF", None, None, 0., 0., None, None, None).is_err());
        assert!(frame_rotation(py, "GCRF", "TEME", Some(2451545.), None, 0., 0., None, None, None).is_err());
        assert!(frame_rotation(py, "GCRF", "LVLH", None, None, 0., 0., None, None, None).is_err());
        assert!(frame_rotation(py, "GCRF", "BODY", None, None, 0., 0., None, None, None).is_err());
        assert!(frame_rotation(py, "GCRF", "invalid", None, None, 0., 0., None, None, None).is_err());
        assert!(frame_rotation(py, "invalid", "GCRF", None, None, 0., 0., None, None, None).is_err());
        assert!(gcrf_to_ecef_matrix(py, f64::NAN, 2451545., 0., 0.).is_ok());
        assert!(gcrf_to_ecef_matrix(py, 2451545., 2451545., f64::INFINITY, 0.).is_ok());
        let v = vec![1., -2., 3.].into_py(py);
        let converted = convert_vector(py, v.as_ref(py), "GCRF", "ECEF", Some(2451545.), Some(2451545.), 0., 0., None, None, None).unwrap();
        let back = convert_vector(py, converted.as_ref(py), "ECEF", "GCRF", Some(2451545.), Some(2451545.), 0., 0., None, None, None).unwrap();
        let values: Vec<f64> = back.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!((values[0]-1.).abs() < 2e-12 && (values[1]+2.).abs() < 2e-12 && (values[2]-3.).abs() < 2e-12);
    });
}

#[test]
fn geodetic_reference_points_and_gcrf_conversion() {
    Python::with_gil(|py| {
        let equator_arg = vec![6378137., 0., 0.].into_py(py);
        let equator = ecef_to_geodetic(py, equator_arg.as_ref(py)).unwrap();
        let values: Vec<f64> = equator.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!(values[0].abs() < 1e-10 && values[1].abs() < 1e-10 && values[2].abs() < 1e-9);
        let pole_arg = vec![0., 0., 6356752.314245].into_py(py);
        let pole = ecef_to_geodetic(py, pole_arg.as_ref(py)).unwrap();
        let values: Vec<f64> = pole.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!((values[0] - 90.).abs() < 1e-8 && values[2].abs() < 1e-6);
        let south_arg = vec![0., 0., -6356752.314245].into_py(py);
        let south = ecef_to_geodetic(py, south_arg.as_ref(py)).unwrap();
        let values: Vec<f64> = south.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert!((values[0] + 90.).abs() < 1e-8 && values[2].abs() < 1e-6);
        for (position, longitude) in [(vec![0., -6378137., 0.], -90.), (vec![-6378137., 0., 0.], 180.)] {
            let arg = position.into_py(py);
            let result = ecef_to_geodetic(py, arg.as_ref(py)).unwrap();
            let values: Vec<f64> = result.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
            assert!((values[1] - longitude).abs() < 1e-10);
        }
        let invalid = vec![f64::INFINITY, 0., 0.].into_py(py);
        assert!(ecef_to_geodetic(py, invalid.as_ref(py)).is_err());
        let direct = gcrf_position_to_geodetic(py, vec![6378137.,0.,0.].into_py(py).as_ref(py), 2451545., 2451545., 0., 0.).unwrap();
        assert_eq!(direct.as_ref(py).getattr("shape").unwrap().extract::<(usize,)>().unwrap(), (3,));
    });
}
