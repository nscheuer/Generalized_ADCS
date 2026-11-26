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
fn keplerian_converts_to_cartesian() {
    Python::with_gil(|py| {
        let a = 7_000_000.0;
        let circular = OrbitalState::from_keplerian(py, a, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
            .unwrap();
        assert!(circular.position.is_none());
        assert!(circular.velocity.is_none());
        let pos: Vec<f64> = circular.position(py).unwrap().extract(py).unwrap();
        let vel: Vec<f64> = circular.velocity(py).unwrap().extract(py).unwrap();
        let expected_v = (crate::orbital_state::EARTH_MU / a).sqrt();
        assert!((pos[0] - a).abs() < 1e-6);
        assert!(pos[1].abs() < 1e-9 && pos[2].abs() < 1e-9);
        assert!(vel[0].abs() < 1e-9 && vel[2].abs() < 1e-9);
        assert!((vel[1] - expected_v).abs() < 1e-9);

        let e = 0.01;
        let inc = 0.2;
        let orbit =
            OrbitalState::from_keplerian(py, a, e, inc, 0.3, 0.4, 0.5, 1.0, None).unwrap();
        assert_eq!(orbit.keplerian_elements.unwrap()[0], a);
        let pos: Vec<f64> = orbit.position(py).unwrap().extract(py).unwrap();
        let vel: Vec<f64> = orbit.velocity(py).unwrap().extract(py).unwrap();
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
        let expected_v2 = crate::orbital_state::EARTH_MU * (2.0 / r - 1.0 / a);
        assert!((v2 - expected_v2).abs() < 1e-6);

        let hx = pos[1] * vel[2] - pos[2] * vel[1];
        let hy = pos[2] * vel[0] - pos[0] * vel[2];
        let hz = pos[0] * vel[1] - pos[1] * vel[0];
        let h = (hx * hx + hy * hy + hz * hz).sqrt();
        let expected_h = (crate::orbital_state::EARTH_MU * a * (1.0 - e * e)).sqrt();
        assert!((h - expected_h).abs() < 1e-3);
        assert!(((hz / h).acos() - inc).abs() < 1e-12);

        let invalid = OrbitalState::from_keplerian(py, a, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0, None)
            .unwrap();
        assert!(invalid.position(py).is_err());
        assert!(invalid.velocity(py).is_err());
    });
}

#[test]
fn pure_rust_keplerian_to_cartesian() {
    let orbit = crate::orbital_state::OrbitalState::new_keplerian(
        [7_000_000.0, 0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2, 0.0, 0.0],
        2451545.0,
    )
    .unwrap();
    let (pos, vel) = orbit.cartesian().unwrap();
    let expected_v = (crate::orbital_state::EARTH_MU / 7_000_000.0).sqrt();
    assert!(pos[0].abs() < 1e-6 && (pos[1] - 7_000_000.0).abs() < 1e-6 && pos[2].abs() < 1e-6);
    assert!(vel[0].abs() < 1e-9 && vel[1].abs() < 1e-9 && (vel[2] - expected_v).abs() < 1e-9);
    assert!(crate::orbital_state::keplerian_to_cartesian(
        [7_000_000.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        crate::orbital_state::EARTH_MU,
    )
    .is_err());
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
