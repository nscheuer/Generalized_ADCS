use super::*;
use pyo3::types::{PyList, PyModule, PyTuple};

fn dependencies(py: Python<'_>, wheel_count: usize, date: f64) -> (PyObject, PyObject, PyObject, PyObject, PyObject, PyObject) {
    let types = PyModule::import(py, "types").unwrap();
    let namespace = types.getattr("SimpleNamespace").unwrap();
    let wheels = PyList::empty(py);
    for _ in 0..wheel_count {
        wheels.append(py.None()).unwrap();
    }
    let adcs = namespace.call1((),).unwrap();
    adcs.setattr("reaction_wheels", wheels).unwrap();
    let satellite = namespace.call0().unwrap();
    satellite.setattr("adcs", adcs).unwrap();
    let environment = namespace.call0().unwrap();
    let record = PyList::new(py, ["state-record"]);
    let state = namespace.call0().unwrap();
    let orbit = namespace.call0().unwrap();
    orbit.setattr("julian_date", date).unwrap();
    state.setattr("orbit", orbit).unwrap();
    state.setattr("record", record).unwrap();
    state.setattr("wheel_count", py.None()).unwrap();
    let scope = pyo3::types::PyDict::new(py);
    scope.set_item("state", state).unwrap();
    let callback = py
        .eval(
            "lambda count: (setattr(state, 'wheel_count', count), state.record)[1]",
            Some(scope),
            Some(scope),
        )
        .unwrap();
    state.setattr("to_record", callback).unwrap();
    let propagator = namespace.call0().unwrap();
    let solver = namespace.call0().unwrap();
    (
        satellite.into_py(py),
        environment.into_py(py),
        state.into_py(py),
        propagator.into_py(py),
        solver.into_py(py),
        record.into_py(py),
    )
}

fn construct(py: Python<'_>, values: &(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject), backend: &str) -> PyResult<SimulationConfig> {
    SimulationConfig::new(
        py,
        values.0.clone_ref(py),
        values.1.clone_ref(py),
        values.2.clone_ref(py),
        values.3.clone_ref(py),
        values.4.clone_ref(py),
        backend,
    )
}

#[test]
fn defaults_preserve_dependencies_and_build_record_with_wheel_count() {
    Python::with_gil(|py| {
        let values = dependencies(py, 2, 2460000.25);
        let config = construct(py, &values, "Python").unwrap();
        assert_eq!(config.backend(), "Python");
        assert!(config.satellite(py).as_ref(py).is(values.0.as_ref(py)));
        assert!(config.environment(py).as_ref(py).is(values.1.as_ref(py)));
        assert!(config.state(py).as_ref(py).is(values.2.as_ref(py)));
        assert!(config.propagator(py).as_ref(py).is(values.3.as_ref(py)));
        assert!(config.solver(py).as_ref(py).is(values.4.as_ref(py)));
        assert!(config.state_record(py).as_ref(py).is(values.5.as_ref(py)));
        assert_eq!(config.julian_date(), 2460000.25);
        assert_eq!(values.2.as_ref(py).getattr("wheel_count").unwrap().extract::<usize>().unwrap(), 2);
    });
}

#[test]
fn all_supported_backends_are_accepted_and_invalid_backend_rejected_before_state_use() {
    Python::with_gil(|py| {
        for backend in ["Python", "C", "Rust", "JAX"] {
            let values = dependencies(py, 0, 2451545.0);
            assert_eq!(construct(py, &values, backend).unwrap().backend(), backend);
        }
        for backend in ["", "python", "CUDA"] {
            let values = dependencies(py, 0, 2451545.0);
            let error = match construct(py, &values, backend) {
                Ok(_) => panic!("unsupported backend unexpectedly accepted"),
                Err(error) => error,
            };
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(values.2.as_ref(py).getattr("wheel_count").unwrap().is_none());
        }
    });
}

#[test]
fn derived_snapshots_are_getter_only_and_julian_date_is_copied() {
    Python::with_gil(|py| {
        let values = dependencies(py, 0, 2451545.0);
        let config = construct(py, &values, "Rust").unwrap();
        assert_eq!(config.state_record(py).as_ref(py).downcast::<PyList>().unwrap().len(), 1);
        values.2.as_ref(py).setattr("orbit", PyTuple::new(py, [2460000.0])).unwrap();
        assert_eq!(config.julian_date(), 2451545.0);

        let py_config = Py::new(py, config).unwrap();
        assert!(py_config.as_ref(py).setattr("state_record", PyList::empty(py)).is_err());
        assert!(py_config.as_ref(py).setattr("julian_date", 2452000.0).is_err());
    });
}

#[test]
fn zero_and_many_wheels_are_passed_as_exact_counts() {
    Python::with_gil(|py| {
        for count in [0, 1, 4] {
            let values = dependencies(py, count, 2451545.0);
            construct(py, &values, "Python").unwrap();
            assert_eq!(
                values.2.as_ref(py).getattr("wheel_count").unwrap().extract::<usize>().unwrap(),
                count
            );
        }
    });
}

#[test]
fn dependency_errors_are_propagated_without_constructing_a_partial_config() {
    Python::with_gil(|py| {
        let values = dependencies(py, 2, 2451545.0);
        // A state lacking the method required to make its record should surface
        // Python's original AttributeError through the Rust binding layer.
        values.2.as_ref(py).delattr("to_record").unwrap();
        let error = match construct(py, &values, "Python") {
            Err(error) => error,
            Ok(_) => panic!("construction should fail when to_record is missing"),
        };
        assert!(error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py));
    });
}

#[test]
fn dependency_properties_are_mutable_and_snapshot_fields_remain_fixed() {
    Python::with_gil(|py| {
        let values = dependencies(py, 1, 2451545.0);
        let mut config = construct(py, &values, "Rust").unwrap();
        config.set_satellite("new-satellite".into_py(py));
        config.set_environment("new-environment".into_py(py));
        config.set_state("new-state".into_py(py));
        config.set_propagator("new-propagator".into_py(py));
        config.set_solver("new-solver".into_py(py));
        config.set_backend("C".to_owned());
        assert_eq!(config.satellite(py).as_ref(py).extract::<String>().unwrap(), "new-satellite");
        assert_eq!(config.environment(py).as_ref(py).extract::<String>().unwrap(), "new-environment");
        assert_eq!(config.state(py).as_ref(py).extract::<String>().unwrap(), "new-state");
        assert_eq!(config.propagator(py).as_ref(py).extract::<String>().unwrap(), "new-propagator");
        assert_eq!(config.solver(py).as_ref(py).extract::<String>().unwrap(), "new-solver");
        assert_eq!(config.backend(), "C");
        assert!(config.state_record(py).as_ref(py).is(values.5.as_ref(py)));
        assert_eq!(config.julian_date(), 2451545.0);
    });
}
