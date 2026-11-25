use crate::sensors::NativeFineSunSensor;

fn make_sensor() -> NativeFineSunSensor {
    NativeFineSunSensor::new(
        [0.0; 3],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [[1e-6, 0.0], [0.0, 2e-6]],
        [0.01, -0.02],
        [0.6, 0.7],
    )
    .unwrap()
}

#[test]
fn measures_sun_boresight_angles_and_adds_bias() {
    let sensor = make_sensor();
    let sun = [0.1, 0.2, 1.0];
    let q = [1.0, 0.0, 0.0, 0.0];
    let actual = sensor.noiseless_biasfree_measurement(sun, q);
    assert!((actual[0] - 0.1_f64.atan2(1.0)).abs() < 1e-12);
    assert!((actual[1] - 0.2_f64.atan2(1.0)).abs() < 1e-12);
    let biased = sensor.noiseless_biased_measurement(sun, q);
    assert!((biased[0] - actual[0] - 0.01).abs() < 1e-12);
    assert!((biased[1] - actual[1] + 0.02).abs() < 1e-12);
}

#[test]
fn availability_and_information_follow_visibility_and_enabled_state() {
    let mut sensor = make_sensor();
    let q = [1.0, 0.0, 0.0, 0.0];
    let sun = [0.0, 0.0, 1.0];
    assert!(sensor.availability(sun, q, true));
    assert!(!sensor.availability(sun, q, false));
    assert!(sensor
        .fisher_information_matrix(sun, q, true)
        .iter()
        .flatten()
        .any(|x| *x != 0.0));
    sensor.enabled = false;
    assert!(!sensor.availability(sun, q, true));
    assert!(sensor
        .fisher_information_matrix(sun, q, true)
        .iter()
        .flatten()
        .all(|x| *x == 0.0));
}
