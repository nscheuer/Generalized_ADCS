use super::*;
use pyo3::types::{PyList, PyTuple};

#[test]
fn geometry_preserves_values_defaults_and_mutable_panel_objects() {
    Python::with_gil(|py| {
        let inertia = PyList::new(py, [
            PyList::new(py, [2.0, 0.0, 0.0]),
            PyList::new(py, [0.0, 3.0, 0.0]),
            PyList::new(py, [0.0, 0.0, 4.0]),
        ]);
        let panel = Py::new(
            py,
            SatellitePanel::new(
                (1, 0, 0).into_py(py),
                (1, 0, 0).into_py(py),
                (2, 1).into_py(py),
                0.0,
                0.0,
                1.0,
                2.2,
            ),
        )
        .unwrap();
        let panel_object = panel.to_object(py);
        let panels = vec![panel_object.clone_ref(py)];
        let center: PyObject = (0.1, -0.2, 0.3).into_py(py);
        let mut geometry = SatelliteGeometry::new(
            py,
            12.5,
            inertia.to_object(py),
            Some(panels),
            Some(center.clone_ref(py)),
        );

        assert_eq!(geometry.mass, 12.5);
        assert!(geometry.inertia.as_ref(py).is(inertia));
        assert!(geometry.center_of_mass.as_ref(py).is(center.as_ref(py)));
        assert_eq!(geometry.panels(py).as_ref(py).downcast::<PyList>().unwrap().len(), 1);
        geometry.mass = 13.0;
        geometry.center_of_mass = (0.25, 0.0, -0.1).into_py(py);
        panel.borrow_mut(py).cd = 2.5;
        assert_eq!(geometry.mass, 13.0);
        assert_eq!(panel.borrow(py).cd, 2.5);

        let defaults = SatelliteGeometry::new(py, 4.0, inertia.to_object(py), None, None);
        assert!(defaults.panels(py).as_ref(py).downcast::<PyList>().unwrap().is_empty());
        assert_eq!(defaults.center_of_mass.as_ref(py).downcast::<PyTuple>().unwrap().len(), 3);
        let first_panels = geometry.panels(py);
        let second_panels = defaults.panels(py);
        assert!(!first_panels.as_ref(py).is(second_panels.as_ref(py)));
    });
}

#[test]
fn geometry_constructor_retains_nested_python_container_references() {
    Python::with_gil(|py| {
        let inertia = PyList::new(py, [1.0, 2.0, 3.0]);
        let geometry = SatelliteGeometry::new(py, 7.0, inertia.to_object(py), None, None);
        inertia.set_item(0, 9.0).unwrap();
        assert_eq!(geometry.inertia.as_ref(py).get_item(0).unwrap().extract::<f64>().unwrap(), 9.0);

        let position = PyList::new(py, [1.0, 2.0, 3.0]);
        let dimensions = PyList::new(py, [4.0, 5.0]);
        let panel = SatellitePanel::new(
            position.to_object(py),
            (0.0, 0.0, 1.0).into_py(py),
            dimensions.to_object(py),
            0.0,
            0.0,
            1.0,
            2.2,
        );
        position.set_item(0, 9.0).unwrap();
        dimensions.append(6.0).unwrap();
        assert_eq!(panel.position.as_ref(py).get_item(0).unwrap().extract::<f64>().unwrap(), 9.0);
        assert_eq!(panel.dimensions.as_ref(py).len().unwrap(), 3);
    });
}
