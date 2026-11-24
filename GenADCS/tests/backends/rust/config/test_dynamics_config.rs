use super::*;
use pyo3::types::PyList;

#[test]
fn defaults_are_empty_and_independent() {
    Python::with_gil(|py| {
        let mut first = DynamicsConfig::new(py, None, None);
        let second = DynamicsConfig::new(py, None, None);
        assert!(first
            .forces(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
        assert!(first
            .torques(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
        first.forces.push("drag".into_py(py));
        first.torques.push("magnetic".into_py(py));
        assert!(second
            .forces(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
        assert!(second
            .torques(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
    });
}

#[test]
fn constructor_and_property_setters_preserve_elements_and_allow_mutation() {
    Python::with_gil(|py| {
        let force = PyList::new(py, ["force-object"]);
        let torque = PyList::new(py, ["torque-object"]);
        let config = DynamicsConfig::new(
            py,
            Some(vec![force.to_object(py)]),
            Some(vec![torque.to_object(py)]),
        );
        let forces = config.forces(py);
        let torques = config.torques(py);
        assert!(forces
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap()
            .is(force));
        assert!(torques
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap()
            .is(torque));

        force.append("changed").unwrap();
        assert_eq!(force.len(), 2);

        let replacement = PyList::new(py, ["replacement"]);
        let mut config = config;
        config.set_forces(vec![replacement.to_object(py)]);
        let exposed = config.forces(py);
        assert!(exposed
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap()
            .is(replacement));
    });
}

#[test]
fn getters_return_new_lists_while_retaining_the_same_python_items() {
    Python::with_gil(|py| {
        let item = PyList::empty(py);
        let config = DynamicsConfig::new(py, Some(vec![item.to_object(py)]), None);
        let first = config.forces(py);
        let second = config.forces(py);
        assert!(!first.as_ref(py).is(second.as_ref(py)));
        assert!(first
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap()
            .is(item));
        first
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .append("external")
            .unwrap();
        assert_eq!(
            config
                .forces(py)
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .len(),
            1
        );
    });
}

#[test]
fn missing_collections_default_independently_and_supplied_collections_are_copied() {
    Python::with_gil(|py| {
        let mut first = DynamicsConfig::new(py, None, None);
        let second = DynamicsConfig::new(py, None, None);
        first.forces.push("drag".into_py(py));
        first.torques.push("magnetic".into_py(py));
        assert!(second.forces.is_empty());
        assert!(second.torques.is_empty());

        let source = PyList::new(py, ["gravity", "srp"]);
        let config = DynamicsConfig::new(
            py,
            Some(source.iter().map(|item| item.into_py(py)).collect()),
            None,
        );
        source.append("later").unwrap();
        let exposed = config.forces(py);
        let exposed = exposed.as_ref(py).downcast::<PyList>().unwrap();
        assert_eq!(exposed.len(), 2);
        assert_eq!(
            exposed.get_item(0).unwrap().extract::<String>().unwrap(),
            "gravity"
        );
    });
}

#[test]
fn setters_replace_collections_without_changing_the_other_collection() {
    Python::with_gil(|py| {
        let mut config = DynamicsConfig::new(
            py,
            Some(vec!["initial-force".into_py(py)]),
            Some(vec!["initial-torque".into_py(py)]),
        );
        config.set_forces(vec!["replacement-force".into_py(py)]);
        let forces = config.forces(py);
        let torques = config.torques(py);
        assert_eq!(
            forces
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .get_item(0)
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "replacement-force"
        );
        assert_eq!(
            torques
                .as_ref(py)
                .downcast::<PyList>()
                .unwrap()
                .get_item(0)
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "initial-torque"
        );

        config.set_torques(Vec::new());
        assert!(config
            .torques(py)
            .as_ref(py)
            .downcast::<PyList>()
            .unwrap()
            .is_empty());
    });
}
