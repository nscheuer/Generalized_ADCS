use super::*;

#[test]
fn cartesian_values_and_validation() {
    Python::with_gil(|py| {
        let p = vec![7e6, 0.0, 0.0].into_py(py);
        let v = vec![0.0, 7500.0, 0.0].into_py(py);
        let orbit =
            OrbitalState::from_cartesian(py, p.as_ref(py), v.as_ref(py), 2451545.0, None).unwrap();
        assert_eq!(orbit.position.unwrap()[0], 7e6);
        assert_eq!(orbit.velocity.unwrap()[1], 7500.0);
        p.as_ref(py).set_item(0, 1.0).unwrap();
        assert_eq!(orbit.position.unwrap()[0], 7e6);
        assert!(
            OrbitalState::from_cartesian(py, p.as_ref(py), v.as_ref(py), f64::INFINITY, None)
                .is_err()
        );
        let bad = vec![f64::NAN, 0.0, 0.0].into_py(py);
        assert!(OrbitalState::from_cartesian(py, bad.as_ref(py), v.as_ref(py), 1.0, None).is_err());
    });
}

#[test]
fn keplerian_has_no_cartesian_conversion() {
    Python::with_gil(|py| {
        let orbit =
            OrbitalState::from_keplerian(py, 7000.0, 0.01, 0.2, 0.3, 0.4, 0.5, 1.0, None).unwrap();
        assert!(orbit.position.is_none());
        assert!(orbit.velocity.is_none());
        assert_eq!(orbit.keplerian_elements.unwrap()[0], 7000.0);
        assert!(orbit.position(py).is_err());
        assert!(orbit.velocity(py).is_err());
    });
}

#[test]
fn setters_preserve_valid_state_after_error() {
    Python::with_gil(|py| {
        let p = vec![1.0, 2.0, 3.0].into_py(py);
        let v = vec![4.0, 5.0, 6.0].into_py(py);
        let mut orbit =
            OrbitalState::from_cartesian(py, p.as_ref(py), v.as_ref(py), 1.0, None).unwrap();
        let invalid = vec![1.0, 2.0].into_py(py);
        assert!(orbit.set_position(py, Some(invalid.as_ref(py))).is_err());
        assert_eq!(orbit.position.unwrap(), [1.0, 2.0, 3.0]);
    });
}
