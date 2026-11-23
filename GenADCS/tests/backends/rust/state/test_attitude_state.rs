use super::*;

#[test]
fn quaternion_normalization_and_owned_values() {
    Python::with_gil(|py| {
        let q = vec![2.0, 0.0, 0.0, 0.0].into_py(py);
        let rates = vec![1.0, 2.0, 3.0].into_py(py);
        let state = AttitudeState::new(py, q.as_ref(py), rates.as_ref(py)).unwrap();
        assert_eq!(state.orientation, [2.0, 0.0, 0.0, 0.0]);
        assert_eq!(state.rates, [1.0, 2.0, 3.0]);
        let normalized: Vec<f64> = state.quaternion_array(py).unwrap().as_ref(py).call_method0("tolist").unwrap().extract().unwrap();
        assert_eq!(normalized, vec![1.0, 0.0, 0.0, 0.0]);
    });
}

#[test]
fn rejects_bad_quaternion_and_rates() {
    Python::with_gil(|py| {
        let good_q = vec![1.0, 0.0, 0.0, 0.0].into_py(py);
        for q in [vec![0.0; 4], vec![f64::NAN, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]] {
            let bad_q = q.into_py(py);
            assert!(AttitudeState::new(py, bad_q.as_ref(py), vec![0.0; 3].into_py(py).as_ref(py)).is_err());
        }
        for rates in [vec![1.0, 2.0], vec![0.0, f64::INFINITY, 0.0]] {
            let rates = rates.into_py(py);
            assert!(AttitudeState::new(py, good_q.as_ref(py), rates.as_ref(py)).is_err());
        }
    });
}

#[test]
fn setters_validate_and_return_new_arrays() {
    Python::with_gil(|py| {
        let q = vec![1.0, 0.0, 0.0, 0.0].into_py(py);
        let rates = vec![0.0; 3].into_py(py);
        let mut state = AttitudeState::new(py, q.as_ref(py), rates.as_ref(py)).unwrap();
        let bad = vec![f64::NAN, 0.0, 0.0].into_py(py);
        assert!(state.set_rates(py, bad.as_ref(py)).is_err());
        assert_eq!(state.rates, [0.0; 3]);
        let array = state.rates(py).unwrap();
        array.as_ref(py).set_item(0, 99.0).unwrap();
        assert_eq!(state.rates[0], 0.0);
    });
}
