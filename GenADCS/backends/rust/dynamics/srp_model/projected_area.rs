pub type Vector3 = [f64; 3];
pub type Quaternion = [f64; 4];

/// Compute projected-area solar radiation pressure force in inertial coordinates.
pub fn projected_area_srp_force(
    position_i: Vector3,
    sun_position_i: Vector3,
    q_b_from_i: Quaternion,
    normals_b: &[Vector3],
    areas: &[f64],
    optical_coefficients: &[Vector3],
    illumination: f64,
) -> Result<Vector3, &'static str> {
    srp_eval(
        position_i,
        sun_position_i,
        q_b_from_i,
        normals_b,
        None,
        areas,
        optical_coefficients,
        illumination,
        false,
    )
}

/// Compute projected-area solar radiation pressure torque in body coordinates.
pub fn projected_area_srp_torque(
    position_i: Vector3,
    sun_position_i: Vector3,
    q_b_from_i: Quaternion,
    normals_b: &[Vector3],
    arms_b: &[Vector3],
    areas: &[f64],
    optical_coefficients: &[Vector3],
    illumination: f64,
) -> Result<Vector3, &'static str> {
    srp_eval(
        position_i,
        sun_position_i,
        q_b_from_i,
        normals_b,
        Some(arms_b),
        areas,
        optical_coefficients,
        illumination,
        true,
    )
}

fn srp_eval(
    position: Vector3,
    sun: Vector3,
    q: Quaternion,
    normals: &[Vector3],
    arms: Option<&[Vector3]>,
    areas: &[f64],
    optical: &[Vector3],
    illumination: f64,
    torque: bool,
) -> Result<Vector3, &'static str> {
    if position.iter().any(|value| !value.is_finite()) {
        return Err("position_i must be a finite 3-vector");
    }
    if sun.iter().any(|value| !value.is_finite()) {
        return Err("sun_position_i must be a finite 3-vector");
    }
    if q.iter().any(|value| !value.is_finite()) {
        return Err("q_b_from_i must be a finite 4-vector");
    }
    if !illumination.is_finite() {
        return Err("illumination must be finite");
    }
    if normals.iter().flatten().any(|value| !value.is_finite()) {
        return Err("normals_b must have 3 finite columns");
    }
    if areas.iter().any(|value| !value.is_finite()) {
        return Err("areas must be finite");
    }
    if optical.iter().flatten().any(|value| !value.is_finite()) {
        return Err("optical_coefficients must have 3 finite columns");
    }
    if arms
        .unwrap_or_default()
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("arms_b must have 3 finite columns");
    }
    if normals.len() != areas.len()
        || optical.len() != areas.len()
        || (torque && arms.map_or(0, |values| values.len()) != areas.len())
    {
        return Err("panel arrays must have matching row counts");
    }

    let delta = [
        sun[0] - position[0],
        sun[1] - position[1],
        sun[2] - position[2],
    ];
    let distance = norm(delta);
    if distance == 0. {
        return Err("spacecraft and Sun positions coincide");
    }
    let rotation = rotation(q);
    let direction = mv(rotation, scale(delta, 1. / distance));
    let pressure = (1361.0 / 299_792_458.0) * (149_597_870_700.0 / distance).powi(2) * illumination;
    let mut total = [0.; 3];
    for i in 0..areas.len() {
        let normal = normals[i];
        let cosine = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
        if cosine > 0. {
            let (specular, diffuse, absorbed) = (optical[i][0], optical[i][1], optical[i][2]);
            let term = add(
                scale(direction, absorbed + diffuse),
                scale(normal, 2. * specular * cosine + 2. * diffuse / 3.),
            );
            let force = scale(term, -pressure * areas[i] * cosine);
            total = add(
                total,
                if torque {
                    cross(arms.expect("arms are required for torque")[i], force)
                } else {
                    force
                },
            );
        }
    }
    if torque {
        Ok(total)
    } else {
        Ok([
            rotation[0][0] * total[0] + rotation[1][0] * total[1] + rotation[2][0] * total[2],
            rotation[0][1] * total[0] + rotation[1][1] * total[1] + rotation[2][1] * total[2],
            rotation[0][2] * total[0] + rotation[1][2] * total[1] + rotation[2][2] * total[2],
        ])
    }
}

fn rotation(q: Quaternion) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
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
    ]
}

fn mv(matrix: [[f64; 3]; 3], vector: Vector3) -> Vector3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: Vector3) -> f64 {
    a.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn add(a: Vector3, b: Vector3) -> Vector3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: Vector3, scalar: f64) -> Vector3 {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
}
