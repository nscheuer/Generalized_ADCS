use super::*;
use pyo3::types::PyList;

#[test]
fn constructor_fields_defaults_and_effect_replacement() {
    Python::with_gil(|py| {
        let orientation = PyList::new(py, [0, 0, 1]);
        let mut wheel = ReactionWheel::new(
            py,
            (1, 0, 0).into_py(py),
            orientation.into_py(py),
            0.02,
            0.1,
            2.0,
            None,
            None,
        )
        .unwrap();
        assert!(wheel.orientation.as_ref(py).is(orientation));
        assert_eq!(
            (
                wheel.rotational_inertia,
                wheel.max_torque,
                wheel.max_momentum
            ),
            (0.02, 0.1, 2.0)
        );
        assert_eq!(
            wheel
                .misalignment
                .as_ref(py)
                .extract::<(f64, f64, f64)>()
                .unwrap(),
            (0.0, 0.0, 0.0)
        );
        assert!(wheel
            .effects(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
        wheel.set_effects(vec![orientation.into_py(py)]);
        assert!(wheel
            .effects(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap()
            .is(orientation));
        wheel.max_torque = 0.3;
        assert_eq!(wheel.max_torque, 0.3);
    });
}

#[test]
fn apply_errors_passes_through_ordered_effects() {
    Python::with_gil(|py| {
        let first = py
            .eval(
                "type('Add', (), {'apply': lambda self, value, dt, rng: value + dt})()",
                None,
                None,
            )
            .unwrap();
        let second = py
            .eval(
                "type('Double', (), {'apply': lambda self, value, dt, rng: value * 2})()",
                None,
                None,
            )
            .unwrap();
        let wheel = ReactionWheel::new(
            py,
            py.None(),
            py.None(),
            0.02,
            0.1,
            2.0,
            None,
            Some(vec![first.into_py(py), second.into_py(py)]),
        )
        .unwrap();
        assert_eq!(
            wheel
                .apply_errors(py, 1.0.into_py(py), 0.5, None)
                .unwrap()
                .as_ref(py)
                .extract::<f64>()
                .unwrap(),
            3.0
        );
    });
}
