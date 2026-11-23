use super::*;

#[test]
fn default_and_owned_momentum() {
    Python::with_gil(|py| {
        assert!(InternalState::new(py, None).unwrap().wheel_momentum.is_empty());
        let source = vec![1.0, -2.0].into_py(py);
        let state = InternalState::new(py, Some(source.as_ref(py))).unwrap();
        assert_eq!(state.wheel_momentum, vec![1.0, -2.0]);
        source.as_ref(py).set_item(0, 99.0).unwrap();
        assert_eq!(state.wheel_momentum[0], 1.0);
    });
}

#[test]
fn rejects_nonfinite_and_nonvector_values() {
    Python::with_gil(|py| {
        for value in [vec![f64::NAN], vec![f64::INFINITY]] {
            let value = value.into_py(py);
            assert!(InternalState::new(py, Some(value.as_ref(py))).is_err());
        }
        let scalar: PyObject = 1.0.into_py(py);
        assert!(InternalState::new(py, Some(scalar.as_ref(py))).is_err());
    });
}

#[test]
fn getter_copies_and_setter_validates() {
    Python::with_gil(|py| {
        let source = vec![1.0].into_py(py);
        let mut state = InternalState::new(py, Some(source.as_ref(py))).unwrap();
        state.wheel_momentum(py).unwrap().as_ref(py).set_item(0, 9.0).unwrap();
        assert_eq!(state.wheel_momentum[0], 1.0);
        let bad = vec![f64::INFINITY].into_py(py);
        assert!(state.set_wheel_momentum(py, bad.as_ref(py)).is_err());
        assert_eq!(state.wheel_momentum[0], 1.0);
    });
}
