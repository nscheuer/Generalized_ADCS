use super::*;

#[test]
fn panel_constructor_preserves_values_defaults_and_allows_mutation() {
    Python::with_gil(|py| {
        let position: PyObject = (1.0, -2.0, 3.5).into_py(py);
        let orientation: PyObject = (0.0, 0.6, 0.8).into_py(py);
        let dimensions: PyObject = (2.5, 1.25).into_py(py);
        let panel = SatellitePanel::new(
            position.clone_ref(py),
            orientation.clone_ref(py),
            dimensions.clone_ref(py),
            0.2,
            0.3,
            0.5,
            2.1,
        );

        assert!(panel.position.as_ref(py).is(position.as_ref(py)));
        assert!(panel.orientation.as_ref(py).is(orientation.as_ref(py)));
        assert!(panel.dimensions.as_ref(py).is(dimensions.as_ref(py)));
        assert_eq!((panel.eta_s, panel.eta_d, panel.eta_a, panel.cd), (0.2, 0.3, 0.5, 2.1));

        let mut default_panel = SatellitePanel::new(
            (0.0, 0.0, 0.0).into_py(py),
            (1.0, 0.0, 0.0).into_py(py),
            (1.0, 2.0).into_py(py),
            0.0,
            0.0,
            1.0,
            2.2,
        );
        default_panel.cd = 2.4;
        assert_eq!(default_panel.cd, 2.4);
        let replacement: PyObject = (1.0, 0.0, 0.0).into_py(py);
        default_panel.position = replacement.clone_ref(py);
        assert!(default_panel.position.as_ref(py).is(replacement.as_ref(py)));
    });
}
