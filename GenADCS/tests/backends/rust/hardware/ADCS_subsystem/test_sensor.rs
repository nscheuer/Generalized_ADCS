use super::NativeSensor;

fn sensor() -> NativeSensor {
    NativeSensor::new(
        &[1.0, 2.0, 3.0],
        &[2.0, 0.0, 0.0, 0.0],
        &[0.0, 3.0, 0.0, 0.0],
        10.0,
        &[
            vec![1.0, 0.1, 0.2],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.3, 3.0],
        ],
        &[0.1, -0.2, 0.3],
    )
    .unwrap()
}

#[test]
fn constructor_preserves_configuration_and_normalizes_quaternions() {
    let sensor = sensor();
    assert_eq!(sensor.position, [1.0, 2.0, 3.0]);
    assert_eq!(sensor.orientation, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(sensor.misalignment, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(sensor.saturation_value, 10.0);
    assert_eq!(sensor.noise[0], [1.0, 0.1, 0.2]);
    assert_eq!(sensor.bias, [0.1, -0.2, 0.3]);
}

#[test]
fn sensors_start_enabled_and_availability_tracks_enabled_state() {
    let mut sensor = sensor();
    assert!(sensor.enabled);
    assert!(sensor.availability());
    sensor.enabled = false;
    assert!(!sensor.availability());
}

#[test]
fn constructor_rejects_incorrect_shapes() {
    let noise = vec![vec![1.0; 3]; 3];
    assert!(NativeSensor::new(&[0.0; 2], &[1.0; 4], &[1.0; 4], 1.0, &noise, &[0.0; 3]).is_err());
    assert!(NativeSensor::new(&[0.0; 3], &[1.0; 3], &[1.0; 4], 1.0, &noise, &[0.0; 3]).is_err());
    assert!(NativeSensor::new(
        &[0.0; 3],
        &[1.0; 4],
        &[1.0; 4],
        1.0,
        &[vec![1.0; 2], vec![1.0; 2], vec![1.0; 2]],
        &[0.0; 3]
    )
    .is_err());
    assert!(NativeSensor::new(&[0.0; 3], &[1.0; 4], &[1.0; 4], 1.0, &noise, &[0.0; 2]).is_err());
}

#[test]
fn constructor_rejects_non_positive_saturation() {
    let noise = vec![vec![1.0; 3]; 3];
    assert!(NativeSensor::new(&[0.0; 3], &[1.0; 4], &[1.0; 4], 0.0, &noise, &[0.0; 3]).is_err());
    assert!(NativeSensor::new(&[0.0; 3], &[1.0; 4], &[1.0; 4], -1.0, &noise, &[0.0; 3]).is_err());
}
