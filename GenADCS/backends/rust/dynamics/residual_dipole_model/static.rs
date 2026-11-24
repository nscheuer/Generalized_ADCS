/// Compute the residual dipole torque in body coordinates.
pub fn static_residual_dipole_torque(
    dipole_b: [f64; 3],
    field_i: [f64; 3],
    q_b_from_i: [f64; 4],
) -> Result<[f64; 3], &'static str> {
    if dipole_b.iter().any(|value| !value.is_finite()) {
        return Err("dipole_b must be a finite 3-vector");
    }
    if field_i.iter().any(|value| !value.is_finite()) {
        return Err("field_i must be a finite 3-vector");
    }
    if q_b_from_i.iter().any(|value| !value.is_finite()) {
        return Err("q_b_from_i must be a finite 4-vector");
    }

    let [w, x, y, z] = q_b_from_i;
    let rotation = [
        [
            w * w + x * x - y * y - z * z,
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let field_b = [
        rotation[0][0] * field_i[0] + rotation[0][1] * field_i[1] + rotation[0][2] * field_i[2],
        rotation[1][0] * field_i[0] + rotation[1][1] * field_i[1] + rotation[1][2] * field_i[2],
        rotation[2][0] * field_i[0] + rotation[2][1] * field_i[1] + rotation[2][2] * field_i[2],
    ];
    Ok([
        dipole_b[1] * field_b[2] - dipole_b[2] * field_b[1],
        dipole_b[2] * field_b[0] - dipole_b[0] * field_b[2],
        dipole_b[0] * field_b[1] - dipole_b[1] * field_b[0],
    ])
}
