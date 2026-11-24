/// Rust-native geometry types and operations shared by the dynamics models.

pub type Vector3 = [f64; 3];
pub type Quaternion = [f64; 4];
pub type Matrix3 = [[f64; 3]; 3];

/// Convert a scalar-first quaternion to its body-from-inertial rotation matrix.
pub fn rotation([w, x, y, z]: Quaternion) -> Matrix3 {
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

/// Multiply a 3 by 3 matrix by a 3-vector.
pub fn mv(a: Matrix3, v: Vector3) -> Vector3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// Compute the cross product of two 3-vectors.
pub fn cross(a: Vector3, b: Vector3) -> Vector3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Compute the Euclidean norm of a 3-vector.
pub fn norm(a: Vector3) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Add two 3-vectors component-wise.
pub fn add(a: Vector3, b: Vector3) -> Vector3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Multiply a 3-vector by a scalar.
pub fn scale(a: Vector3, s: f64) -> Vector3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
