use super::*;
use pyo3::types::PyList;

#[test]
fn defaults_are_empty_and_independent() {
    Python::with_gil(|py| {
        let mut first = DynamicsConfig::new(None, None);
        let second = DynamicsConfig::new(None, None);
        assert!(first.forces(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        assert!(first.torques(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        first.forces.push("drag".into_py(py));
        first.torques.push("magnetic".into_py(py));
        assert!(second.forces(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        assert!(second.torques(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
    });
}

#[test]
fn constructor_and_property_setters_preserve_elements_and_allow_mutation() {
    Python::with_gil(|py| {
        let force = PyList::new(py, ["force-object"]);
        let torque = PyList::new(py, ["torque-object"]);
        let config = DynamicsConfig::new(
            Some(vec![force.to_object(py)]),
            Some(vec![torque.to_object(py)]),
        );
        let forces = config.forces(py);
        let torques = config.torques(py);
        assert!(forces.as_ref(py).downcast::<PyList>().unwrap().get_item(0).unwrap().is(force));
        assert!(torques.as_ref(py).downcast::<PyList>().unwrap().get_item(0).unwrap().is(torque));

        force.append("changed").unwrap();
        assert_eq!(force.len().unwrap(), 2);

        let replacement = PyList::new(py, ["replacement"]);
        let mut config = config;
        config.set_forces(vec![replacement.to_object(py)]);
        let exposed = config.forces(py);
        assert!(exposed.as_ref(py).downcast::<PyList>().unwrap().get_item(0).unwrap().is(replacement));
    });
}

#[test]
fn getters_return_new_lists_while_retaining_the_same_python_items() {
    Python::with_gil(|py| {
        let item = PyList::empty(py);
        let config = DynamicsConfig::new(Some(vec![item.to_object(py)]), None);
        let first = config.forces(py);
        let second = config.forces(py);
        assert!(!first.as_ref(py).is(second.as_ref(py)));
        assert!(first.as_ref(py).downcast::<PyList>().unwrap().get_item(0).unwrap().is(item));
        first.as_ref(py).downcast::<PyList>().unwrap().append("external").unwrap();
        assert_eq!(config.forces(py).as_ref(py).downcast::<PyList>().unwrap().len(), 1);
    });
}
