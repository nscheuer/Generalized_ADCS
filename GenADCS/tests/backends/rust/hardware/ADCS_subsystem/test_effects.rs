use super::*;

#[test]
fn effects_run_in_order_and_receive_dt_and_rng() {
    Python::with_gil(|py| {
        let module = py.import("types").unwrap();
        let namespace = module.getattr("SimpleNamespace").unwrap();
        let effect = namespace.call0().unwrap();
        effect
            .setattr(
                "apply",
                py.eval("lambda value, dt, rng: value + dt + rng", None, None)
                    .unwrap(),
            )
            .unwrap();
        let result = apply_effects(
            py,
            1.0.into_py(py),
            vec![effect.into_py(py), effect.into_py(py)],
            0.5,
            Some(2.0.into_py(py)),
        )
        .unwrap();
        assert_eq!(result.as_ref(py).extract::<f64>().unwrap(), 6.0);
    });
}

#[test]
fn empty_effects_preserve_original_object_and_errors_propagate() {
    Python::with_gil(|py| {
        let value = py.eval("object()", None, None).unwrap().into_py(py);
        assert!(apply_effects(py, value.clone_ref(py), vec![], 1.0, None)
            .unwrap()
            .as_ref(py)
            .is(value.as_ref(py)));
        let effect = py
            .eval(
                "type('Broken', (), {'apply': lambda self, value, dt, rng: 1 / 0})()",
                None,
                None,
            )
            .unwrap();
        assert!(apply_effects(py, value, vec![effect.into_py(py)], 1.0, None).is_err());
    });
}
