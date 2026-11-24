use super::SatellitePanel;

#[test]
fn panel_constructor_sets_native_values_and_defaults() {
    let panel = SatellitePanel::new([1.0, -2.0, 3.5], [0.0, 0.6, 0.8], [2.5, 1.25]);

    assert_eq!(panel.position, [1.0, -2.0, 3.5]);
    assert_eq!(panel.orientation, [0.0, 0.6, 0.8]);
    assert_eq!(panel.dimensions, [2.5, 1.25]);
    assert_eq!(
        (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd),
        (0.0, 0.0, 1.0, 2.2)
    );
}
