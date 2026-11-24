/// Compute the combined solar and lunar third-body acceleration in the
/// central body's inertial frame.
pub fn lunisolar_acceleration_native(
    position_i: [f64; 3],
    sun_position_i: [f64; 3],
    moon_position_i: [f64; 3],
    sun_mu: f64,
    moon_mu: f64,
) -> Result<[f64; 3], &'static str> {
    if position_i
        .iter()
        .chain(sun_position_i.iter())
        .chain(moon_position_i.iter())
        .chain([sun_mu, moon_mu].iter())
        .any(|value| !value.is_finite())
    {
        return Err("third-body inputs must be finite");
    }

    let mut result = [0.; 3];
    for (body, mu) in [(sun_position_i, sun_mu), (moon_position_i, moon_mu)] {
        let rel = [
            body[0] - position_i[0],
            body[1] - position_i[1],
            body[2] - position_i[2],
        ];
        let d = norm(body);
        let r = norm(rel);
        if d == 0. || r == 0. {
            return Err("third-body distance must be nonzero");
        }
        result = add(
            result,
            scale(
                [
                    rel[0] / r.powi(3) - body[0] / d.powi(3),
                    rel[1] / r.powi(3) - body[1] / d.powi(3),
                    rel[2] / r.powi(3) - body[2] / d.powi(3),
                ],
                mu,
            ),
        );
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Err("third-body acceleration must be finite");
    }
    Ok(result)
}
