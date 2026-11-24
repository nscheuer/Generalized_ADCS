use super::{SatelliteGeometry, SatellitePanel};

#[test]
fn geometry_constructor_sets_native_defaults() {
    let mut geometry =
        SatelliteGeometry::new(12.5, [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);

    assert_eq!(geometry.mass, 12.5);
    assert_eq!(
        geometry.inertia,
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    );
    assert!(geometry.panels.is_empty());
    assert_eq!(geometry.center_of_mass, [0.0; 3]);

    geometry
        .panels
        .push(SatellitePanel::new([1.0, 0.0, 0.0], [0.0; 3], [2.0, 1.0]));
    geometry.center_of_mass = [0.1, -0.2, 0.3];
    assert_eq!(geometry.panels.len(), 1);
    assert_eq!(geometry.center_of_mass, [0.1, -0.2, 0.3]);
}
