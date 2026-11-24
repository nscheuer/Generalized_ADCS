//! Native projected-area drag force and torque calculations.

pub type Vector3 = [f64; 3];
pub type Quaternion = [f64; 4];

fn rotation([w, x, y, z]: Quaternion) -> [[f64; 3]; 3] {
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

fn mv(a: [[f64; 3]; 3], v: Vector3) -> Vector3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Compute inertial-frame drag force and body-frame drag torque.
pub fn projected_area_drag(
    relative_velocity_i: Vector3,
    q_b_from_i: Quaternion,
    density_kg_m3: f64,
    normals_b: &[Vector3],
    arms_b: Option<&[Vector3]>,
    areas_m2: &[f64],
    drag_coefficients: &[f64],
) -> Result<(Vector3, Vector3), &'static str> {
    if !relative_velocity_i
        .iter()
        .chain(q_b_from_i.iter())
        .chain(areas_m2)
        .chain(drag_coefficients)
        .chain(std::iter::once(&density_kg_m3))
        .all(|v| v.is_finite())
        || normals_b.iter().flatten().any(|v| !v.is_finite())
        || arms_b
            .unwrap_or(&[])
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err("drag inputs must be finite");
    }
    if normals_b.len() != areas_m2.len()
        || drag_coefficients.len() != areas_m2.len()
        || arms_b.is_some_and(|arms| arms.len() != areas_m2.len())
    {
        return Err("panel arrays must have matching row counts");
    }
    let rot = rotation(q_b_from_i);
    let velocity = mv(rot, relative_velocity_i);
    let speed = velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut force_b = [0.; 3];
    let mut torque_b = [0.; 3];
    if speed > 0. && density_kg_m3 > 0. {
        for i in 0..areas_m2.len() {
            let incidence = normals_b[i]
                .iter()
                .zip(velocity)
                .map(|(n, v)| n * v)
                .sum::<f64>()
                / speed;
            if incidence > 0. {
                let factor =
                    -0.5 * density_kg_m3 * speed * areas_m2[i] * drag_coefficients[i] * incidence;
                let f = [
                    velocity[0] * factor,
                    velocity[1] * factor,
                    velocity[2] * factor,
                ];
                force_b = [force_b[0] + f[0], force_b[1] + f[1], force_b[2] + f[2]];
                if let Some(arms) = arms_b {
                    let t = cross(arms[i], f);
                    torque_b = [torque_b[0] + t[0], torque_b[1] + t[1], torque_b[2] + t[2]];
                }
            }
        }
    }
    let force_i = [
        rot[0][0] * force_b[0] + rot[1][0] * force_b[1] + rot[2][0] * force_b[2],
        rot[0][1] * force_b[0] + rot[1][1] * force_b[1] + rot[2][1] * force_b[2],
        rot[0][2] * force_b[0] + rot[1][2] * force_b[1] + rot[2][2] * force_b[2],
    ];
    Ok((force_i, torque_b))
}
