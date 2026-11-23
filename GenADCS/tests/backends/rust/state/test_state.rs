use super::*;

#[test]
fn record_roundtrip_and_wheel_count() {
    Python::with_gil(|py| {
        let attitude = Py::new(py, crate::AttitudeState { orientation: [2.0, 0.0, 0.0, 0.0], rates: [0.1, 0.2, 0.3] }).unwrap();
        let p = vec![7e6, 0.0, 0.0].into_py(py);
        let v = vec![0.0, 7500.0, 0.0].into_py(py);
        let orbit = Py::new(py, OrbitalState::from_cartesian(py, p.as_ref(py), v.as_ref(py), 2451545.0, None).unwrap()).unwrap();
        let internal = Py::new(py, InternalState { wheel_momentum: vec![0.1, -0.2] }).unwrap();
        let state = State::new(py, attitude, orbit, Some(internal)).unwrap();
        assert!(state.to_record(py, Some(2)).is_ok());
        assert!(state.to_record(py, Some(1)).is_err());
        assert!(state.to_record(py, Some(-1)).is_err());
        let record = state.to_record(py, None).unwrap();
        let rebuilt = State::from_record(py, record.as_ref(py), 2451545.0, None).unwrap();
        assert_eq!(rebuilt.internal.borrow(py).wheel_momentum, vec![0.1, -0.2]);
        assert_eq!(rebuilt.attitude.borrow(py).orientation, [1.0, 0.0, 0.0, 0.0]);
    });
}
