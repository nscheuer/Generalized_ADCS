use super::NativeADCSSubsystem;

#[test]
fn native_component_collections_are_independent_and_replaceable() {
    let mut first = NativeADCSSubsystem::new(vec!["wheel"], Vec::<&str>::new());
    let second = NativeADCSSubsystem::<&str, &str>::default();

    first.reaction_wheels = vec!["replacement"];
    first.magnetorquers.push("torquer");

    assert_eq!(first.reaction_wheels, ["replacement"]);
    assert_eq!(first.magnetorquers, ["torquer"]);
    assert!(second.reaction_wheels.is_empty());
    assert!(second.magnetorquers.is_empty());
    assert!(second.sensors.is_empty());
    assert_eq!(second.N, 1);
    assert!(second.estimator.is_none());
}

#[test]
fn native_subsystem_accepts_sensors_estimator_and_positive_substeps() {
    let system = NativeADCSSubsystem::with_components(
        vec!["wheel"],
        vec!["torquer"],
        vec!["gyro", "tracker"],
        4,
        Some("mekf"),
    )
    .unwrap();
    assert_eq!(system.reaction_wheels, ["wheel"]);
    assert_eq!(system.magnetorquers, ["torquer"]);
    assert_eq!(system.sensors, ["gyro", "tracker"]);
    assert_eq!(system.N, 4);
    assert_eq!(system.estimator, Some("mekf"));
    assert!(NativeADCSSubsystem::with_components::<&str, &str, &str, &str>(
        vec![], vec![], vec![], 0, None
    )
    .is_err());
}
