//! Native static gravity-gradient torque calculation.

pub type Vector3 = [f64; 3];
pub type Quaternion = [f64; 4];
pub type Matrix3 = [[f64; 3]; 3];

fn rotation([w, x, y, z]: Quaternion) -> Matrix3 {
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

fn mv(a: Matrix3, v: Vector3) -> Vector3 {
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

/// Compute gravity-gradient torque in body coordinates.
pub fn static_gravity_gradient_torque(
    position_i: Vector3,
    q_b_from_i: Quaternion,
    inertia_b: Matrix3,
    gravitational_parameter: f64,
) -> Result<Vector3, &'static str> {
    if !position_i
        .iter()
        .chain(q_b_from_i.iter())
        .chain(inertia_b.iter().flatten())
        .chain(std::iter::once(&gravitational_parameter))
        .all(|v| v.is_finite())
    {
        return Err("gravity-gradient inputs must be finite");
    }
    let r = position_i.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r == 0. {
        return Err("gravity gradient is undefined at the body's centre");
    }
    let radial = mv(
        rotation(q_b_from_i),
        [position_i[0] / r, position_i[1] / r, position_i[2] / r],
    );
    let ir = mv(inertia_b, radial);
    let t = cross(radial, ir);
    let scale = 3. * gravitational_parameter / r.powi(3);
    Ok([t[0] * scale, t[1] * scale, t[2] * scale])
}
