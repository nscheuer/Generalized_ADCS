pub fn quaternion_derivative_native(q: [f64; 4], w: [f64; 3]) -> [f64; 4] {
    [
        0.5 * (q[1] * w[0] + q[2] * w[1] + q[3] * w[2]),
        -0.5 * (q[0] * w[0] + w[1] * q[3] - w[2] * q[2]),
        -0.5 * (q[0] * w[1] + w[2] * q[1] - w[0] * q[3]),
        -0.5 * (q[0] * w[2] + w[0] * q[2] - w[1] * q[1]),
    ]
}

pub fn quaternion_rate_matrix_native([w, x, y, z]: [f64; 4]) -> [[f64; 3]; 4] {
    [
        [x / 2.0, y / 2.0, z / 2.0],
        [-w / 2.0, -z / 2.0, y / 2.0],
        [z / 2.0, -w / 2.0, -x / 2.0],
        [-y / 2.0, x / 2.0, -w / 2.0],
    ]
}

pub fn angular_velocity_from_quaternion_derivative_native(q: [f64; 4], qd: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = q;
    let [a, b, c, d] = qd;
    [
        2.0 * (x * a - w * b + z * c - y * d),
        2.0 * (y * a - z * b - w * c + x * d),
        2.0 * (z * a + y * b - x * c - w * d),
    ]
}

pub fn quaternion_normalization_jacobian_native(
    q: [f64; 4],
) -> Result<[[f64; 4]; 4], &'static str> {
    let n = qnorm(q);
    if n == 0.0 {
        return Err("quaternion must have nonzero norm");
    }
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| (if i == j { 1.0 / n } else { 0.0 }) - q[i] * q[j] / n.powi(3))
    }))
}

pub fn rotated_vector_quaternion_jacobian_native(q: [f64; 4], v: [f64; 3]) -> [[f64; 4]; 3] {
    let u = [q[1], q[2], q[3]];
    let c = cross3(u, v);
    let dotuv = dot(&u, &v);
    let sv = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if j == 0 {
                2.0 * c[i]
            } else {
                let k = j - 1;
                2.0 * (if i == k { dotuv } else { 0.0 } + u[i] * v[k]
                    - 2.0 * v[i] * u[k]
                    - q[0] * sv[i][k])
            }
        })
    })
}

pub fn quaternion_to_euler321_native([w, x, y, z]: [f64; 4]) -> [f64; 3] {
    [
        f64::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        f64::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    ]
}
