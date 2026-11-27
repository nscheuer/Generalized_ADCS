use super::*;
use pyo3::types::PyList;

#[test]
fn constructor_fields_defaults_and_noise_bias_validation() {
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
            0.0,
            0.0,
            None,
        )
        .unwrap();
        assert!(wheel.orientation.as_ref(py).is(orientation));
        assert_eq!(
            (
                wheel.rotational_inertia,
                wheel.max_torque,
                wheel.max_momentum,
                wheel.noise(),
                wheel.bias(),
            ),
            (0.02, 0.1, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            wheel
                .misalignment
                .as_ref(py)
                .extract::<(f64, f64, f64)>()
                .unwrap(),
            (0.0, 0.0, 0.0)
        );
        wheel.set_noise(0.05).unwrap();
        wheel.set_bias(-0.01).unwrap();
        assert_eq!(wheel.noise(), 0.05);
        assert_eq!(wheel.bias(), -0.01);
        assert!(wheel.set_noise(-0.1).is_err());
        wheel.max_torque = 0.3;
        assert_eq!(wheel.max_torque, 0.3);
    });
}
