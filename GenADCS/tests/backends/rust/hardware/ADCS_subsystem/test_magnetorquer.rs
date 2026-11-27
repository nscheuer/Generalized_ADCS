use super::*;
use pyo3::types::PyList;

#[test]
fn defaults_mutability_and_noise_bias_validation() {
    Python::with_gil(|py| {
        let position = PyList::new(py, [1, 2, 3]);
        let mut torquer = Magnetorquer::new(
            py,
            position.into_py(py),
            (0, 0, 1).into_py(py),
            0.2,
            None,
            0.0,
            0.0,
            None,
        )
        .unwrap();
        assert!(torquer.position.as_ref(py).is(position));
        assert_eq!(torquer.max_dipole, 0.2);
        assert_eq!(torquer.noise(), 0.0);
        assert_eq!(torquer.bias(), 0.0);
        assert_eq!(
            torquer
                .misalignment
                .as_ref(py)
                .extract::<(f64, f64, f64)>()
                .unwrap(),
            (0.0, 0.0, 0.0)
        );
        torquer.max_dipole = 0.3;
        torquer.set_noise(0.02).unwrap();
        torquer.set_bias(0.05).unwrap();
        assert_eq!(torquer.max_dipole, 0.3);
        assert_eq!(torquer.noise(), 0.02);
        assert_eq!(torquer.bias(), 0.05);
        assert!(torquer.set_noise(-1.0).is_err());
    });
}
