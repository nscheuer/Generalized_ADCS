use crate::orbital_propagators::keplerian_orbital_propagator::keplerian_step_native;

#[test]
fn circular_orbit_full_period_returns_to_initial_state() {
    let mu = 3.986_004_418e14;
    let radius: f64 = 7_000_000.0;
    let speed = (mu / radius).sqrt();
    let period = 2.0 * std::f64::consts::PI * (radius.powi(3) / mu).sqrt();

    let (pos, vel) =
        keplerian_step_native([radius, 0.0, 0.0], [0.0, speed, 0.0], mu, period).unwrap();
    assert!((pos[0] - radius).abs() < 1e-5);
    assert!(pos[1].abs() < 1e-5);
    assert!(vel[0].abs() < 1e-8);
    assert!((vel[1] - speed).abs() < 1e-8);
}

#[test]
fn rejects_zero_or_nonfinite_radius() {
    assert!(keplerian_step_native([0.0; 3], [0.0; 3], 3.986e14, 1.0).is_err());
    assert!(keplerian_step_native([f64::NAN, 0.0, 0.0], [0.0; 3], 3.986e14, 1.0).is_err());
}
