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
}
