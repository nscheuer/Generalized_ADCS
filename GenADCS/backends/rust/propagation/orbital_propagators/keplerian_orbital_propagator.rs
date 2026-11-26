fn stumpff(z: f64) -> (f64, f64) {
    if z > 1e-8 {
        let root = z.sqrt();
        ((1.0 - root.cos()) / z, (root - root.sin()) / root.powi(3))
    } else if z < -1e-8 {
        let root = (-z).sqrt();
        (
            (root.cosh() - 1.0) / -z,
            (root.sinh() - root) / root.powi(3),
        )
    } else {
        (
            0.5 - z / 24.0 + z * z / 720.0,
            1.0 / 6.0 - z / 120.0 + z * z / 5040.0,
        )
    }
}

pub fn keplerian_step_native(
    position: [f64; 3],
    velocity: [f64; 3],
    mu: f64,
    dt: f64,
) -> Result<([f64; 3], [f64; 3]), String> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let radius = dot(position, position).sqrt();
    if radius <= 0.0 || !radius.is_finite() {
        return Err("orbital position must have positive radius".into());
    }
    let root_mu = mu.sqrt();
    let radial_speed = dot(position, velocity) / radius;
    let alpha = 2.0 / radius - dot(velocity, velocity) / mu;
    let mut chi = root_mu
        * dt
        * if alpha.abs() > 1e-12 {
            alpha.abs()
        } else {
            1.0 / radius
        };
    let mut converged = false;
    for _ in 0..100 {
        let z = alpha * chi * chi;
        let (c, s) = stumpff(z);
        let residual = radius * radial_speed / root_mu * chi * chi * c
            + (1.0 - alpha * radius) * chi.powi(3) * s
            + radius * chi
            - root_mu * dt;
        let derivative = radius * radial_speed / root_mu * chi * (1.0 - z * s)
            + (1.0 - alpha * radius) * chi * chi * c
            + radius;
        let change = residual / derivative;
        chi -= change;
        if change.abs() <= 1e-12 * 1.0_f64.max(chi.abs()) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("Keplerian universal anomaly did not converge".into());
    }
    let z = alpha * chi * chi;
    let (c, s) = stumpff(z);
    let f = 1.0 - chi * chi * c / radius;
    let g = dt - chi.powi(3) * s / root_mu;
    let next_position = std::array::from_fn(|i| f * position[i] + g * velocity[i]);
    let next_radius = dot(next_position, next_position).sqrt();
    let fdot = root_mu / (next_radius * radius) * (z * s - 1.0) * chi;
    let gdot = 1.0 - chi * chi * c / next_radius;
    let next_velocity = std::array::from_fn(|i| fdot * position[i] + gdot * velocity[i]);
    Ok((next_position, next_velocity))
}

#[cfg(test)]
mod tests {
    use super::keplerian_step_native;

    #[test]
    fn circular_orbit_quarter_period_matches_expected_position() {
        let mu = 3.986_004_418e14;
        let radius: f64 = 7_000_000.0;
        let speed = (mu / radius).sqrt();
        let period = 2.0 * std::f64::consts::PI * (radius.powi(3) / mu).sqrt();
        let (position, velocity) =
            keplerian_step_native([radius, 0.0, 0.0], [0.0, speed, 0.0], mu, period / 4.0).unwrap();
        assert!((position[0]).abs() < 1e-5);
        assert!((position[1] - radius).abs() < 1e-5);
        assert!((velocity[0] + speed).abs() < 1e-8);
        assert!(velocity[1].abs() < 1e-8);
    }

    #[test]
    fn rejects_zero_radius() {
        assert!(keplerian_step_native([0.0; 3], [0.0; 3], 1.0, 1.0).is_err());
    }
}
