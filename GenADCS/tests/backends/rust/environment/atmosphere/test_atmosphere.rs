use super::*;

#[test]
fn corotating_velocity_matches_rigid_rotation_and_custom_rate() {
    let position = [7_000_000.0, 0.0, 0.0];
    let velocity = native_corotation(position);
    assert_eq!(velocity[0], 0.0);
    assert!((velocity[1] - 7.2921150e-5 * position[0]).abs() < 1e-12);
    assert_eq!(velocity[2], 0.0);

    let custom = [4.0, 5.0, 6.0];
    let p = [1.0, 2.0, 3.0];
    let cross = [
        custom[1] * p[2] - custom[2] * p[1],
        custom[2] * p[0] - custom[0] * p[2],
        custom[0] * p[1] - custom[1] * p[0],
    ];
    Python::with_gil(|py| {
        let py_position = p.into_py(py);
        let actual =
            co_rotating_atmosphere_velocity(py, py_position.as_ref(py), 0.0, Some(custom)).unwrap();
        let actual: Vec<f64> = actual
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        for i in 0..3 {
            assert!((actual[i] - cross[i]).abs() < 1e-14);
        }
    });
}

#[test]
fn nrlmsise_density_is_realistic_and_decreases_with_altitude() {
    let epoch = 2_458_849.5;
    let low = native_density([6_778_137.0, 0.0, 0.0], epoch).unwrap();
    let high = native_density([7_000_000.0, 0.0, 0.0], epoch).unwrap();
    assert!(
        low.is_finite() && 1e-13 < low && low < 1e-9,
        "density={low}"
    );
    assert!(low > high && high > 0.0, "low={low}, high={high}");
}

#[test]
fn nrlmsise_rejects_invalid_ap_history() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_atmosphere").unwrap();
        _atmosphere(py, module).unwrap();
        let model = module.getattr("NRLMSISE00").unwrap();
        let position = vec![6_778_137.0, 0.0, 0.0];
        let ap = vec![1.0; 6];
        let result = model.call_method1(
            "density",
            (position, 2_458_849.5, 150.0, 150.0, 4.0, true, ap),
        );
        assert!(result.is_err());
    });
}

#[test]
fn generic_atmospheric_density_is_explicitly_unimplemented() {
    Python::with_gil(|py| {
        let position = vec![7_000_000.0, 0.0, 0.0].into_py(py);
        let error = atmospheric_density(py, position.as_ref(py), 2_451_545.0).unwrap_err();
        assert!(error.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
    });
}
