use super::*;
use pyo3::types::PyList;

#[test]
fn independent_defaults_and_replaced_collections() {
    Python::with_gil(|py| {
        let mut first = ADCSSubsystem::new(None, None);
        let second = ADCSSubsystem::new(None, None);
        first.set_reaction_wheels(vec!["wheel".into_py(py)]);
        assert!(second.reaction_wheels(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        assert!(first.magnetorquers(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        first.set_magnetorquers(vec!["torquer".into_py(py)]);
        assert_eq!(first.magnetorquers(py).as_ref(py).downcast::<PyList>().unwrap().len(), 1);
    });
}

#[test]
fn getters_copy_lists_but_keep_item_identity() {
    Python::with_gil(|py| {
        let wheel = PyList::empty(py);
        let system = ADCSSubsystem::new(Some(vec![wheel.into_py(py)]), None);
        let a = system.reaction_wheels(py);
        let b = system.reaction_wheels(py);
        assert!(!a.as_ref(py).is(b.as_ref(py)));
        let list = a.as_ref(py).downcast::<PyList>().unwrap();
        assert!(list.get_item(0).unwrap().is(wheel));
        list.append("local change").unwrap();
        assert_eq!(system.reaction_wheels(py).as_ref(py).downcast::<PyList>().unwrap().len(), 1);
    });
}
