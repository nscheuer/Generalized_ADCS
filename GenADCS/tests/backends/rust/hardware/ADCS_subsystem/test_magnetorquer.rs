use super::*;
use pyo3::types::PyList;

#[test]
fn defaults_mutability_and_effect_list_ownership() {
    Python::with_gil(|py| {
        let position = PyList::new(py, [1, 2, 3]);
        let mut torquer = Magnetorquer::new(
            py,
            position.into_py(py),
            (0, 0, 1).into_py(py),
            0.2,
            None,
            None,
        );
        assert!(torquer.position.as_ref(py).is(position));
        assert_eq!(torquer.max_dipole, 0.2);
        assert_eq!(
            torquer
                .misalignment
                .as_ref(py)
                .extract::<(f64, f64, f64)>()
                .unwrap(),
            (0.0, 0.0, 0.0)
        );
        torquer.max_dipole = 0.3;
        torquer.set_effects(vec![position.into_py(py)]);
        let exposed = torquer.effects(py);
        exposed
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .append("local")
            .unwrap();
        assert_eq!(
            torquer
                .effects(py)
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(torquer.max_dipole, 0.3);
    });
}

#[test]
fn filters_and_apply_errors_use_the_stored_effects() {
    Python::with_gil(|py| {
        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../")
            .canonicalize()
            .unwrap();
        py.import("sys")
            .unwrap()
            .getattr("path")
            .unwrap()
            .call_method1("insert", (0, project_root.to_str().unwrap()))
            .unwrap();
        let bias = py
            .import("GenADCS.backends.python.bias")
            .unwrap()
            .getattr("FixedBias")
            .unwrap()
            .call1((2.0,))
            .unwrap();
        let noise = py
            .import("GenADCS.backends.python.noise")
            .unwrap()
            .getattr("GaussianWhiteNoise")
            .unwrap()
            .call1((0.0,))
            .unwrap();
        let effect = py
            .eval(
                "type('Effect', (), {'apply': lambda self, value, dt, rng: value + dt})()",
                None,
                None,
            )
            .unwrap();
        let mut torquer = Magnetorquer::new(
            py,
            py.None(),
            py.None(),
            0.2,
            None,
            Some(vec![
                bias.into_py(py),
                noise.into_py(py),
                effect.into_py(py),
            ]),
        );
        assert_eq!(
            torquer
                .biases(py)
                .unwrap()
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            torquer
                .noises(py)
                .unwrap()
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .len(),
            1
        );
        let effect = py
            .eval(
                "type('Effect', (), {'apply': lambda self, value, dt, rng: value + dt})()",
                None,
                None,
            )
            .unwrap();
        torquer.set_effects(vec![effect.into_py(py)]);
        assert_eq!(
            torquer
                .apply_errors(py, 1.0.into_py(py), 0.5, None)
                .unwrap()
                .as_ref(py)
                .extract::<f64>()
                .unwrap(),
            1.5
        );
    });
}
