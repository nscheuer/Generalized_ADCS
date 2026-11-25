//! Rust implementations of the shared sensor measurement and estimation kernels.

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];
pub type Vec2 = [f64; 2];
pub type Mat2 = [[f64; 2]; 2];

pub fn cholesky3(a: Mat3) -> Option<Mat3> {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut v = a[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(v > 0.0) {
                    return None;
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    Some(l)
}

pub fn vector_measurement(
    v: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    bias: Vec3,
    l: Mat3,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    draw: Option<Vec3>,
) -> Vec3 {
    let mut out = mul3v(mounting(orientation, misalignment), rotate(attitude, v));
    if with_bias {
        out = add3(out, bias);
    }
    if with_noise {
        out = add3(out, mul3v(l, draw.unwrap_or_else(|| normals3(0, 0))));
    }
    out.map(|x| x.max(-saturation).min(saturation))
}

pub fn rate_measurement(
    rate: Vec3,
    orientation: [f64; 4],
    misalignment: [f64; 4],
    bias: Vec3,
    l: Mat3,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    draw: Option<Vec3>,
) -> Vec3 {
    let mut out = mul3v(mounting(orientation, misalignment), rate);
    if with_bias {
        out = add3(out, bias)
    }
    if with_noise {
        out = add3(out, mul3v(l, draw.unwrap_or_else(|| normals3(0, 0))))
    }
    out.map(|x| x.max(-saturation).min(saturation))
}

pub fn scalar_sun_measurement(
    sun: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    normal: Vec3,
    sensitivity: f64,
    bias: f64,
    noise_std: f64,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    draw: Option<f64>,
) -> f64 {
    let body = rotate(attitude, sun);
    let normal_body = mul3v(transpose(mounting(orientation, misalignment)), normal);
    let mut out = sensitivity * dot(normal_body, body).max(0.);
    if with_bias {
        out += bias
    }
    if with_noise {
        out += noise_std * draw.unwrap_or_else(|| normals3(0, 0)[0])
    }
    out.max(0.).min(saturation)
}

pub fn fine_direction_measurement(
    direction: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    bias: Vec2,
    l: Mat2,
    with_bias: bool,
    with_noise: bool,
    draw: Option<Vec2>,
) -> Vec2 {
    let s = mul3v(
        mounting(orientation, misalignment),
        rotate(attitude, direction),
    );
    let mut out = [s[0].atan2(s[2]), s[1].atan2(s[2])];
    if with_bias {
        out = add2(out, bias)
    }
    if with_noise {
        out = add2(
            out,
            mul2v(
                l,
                draw.unwrap_or_else(|| [normals3(0, 0)[0], normals3(0, 0)[1]]),
            ),
        )
    }
    out
}

pub fn gnss_measurement(
    position: [f64; 3],
    velocity: [f64; 3],
    bias: Vec<f64>,
    l: Vec<Vec<f64>>,
    with_bias: bool,
    with_noise: bool,
    draw: Option<Vec<f64>>,
) -> Vec<f64> {
    let mut out = position.into_iter().chain(velocity).collect::<Vec<_>>();
    if with_bias {
        for (x, b) in out.iter_mut().zip(bias) {
            *x += b
        }
    }
    if with_noise {
        let z = draw.unwrap_or_else(|| normals(0, 0, out.len()));
        for i in 0..out.len() {
            for j in 0..=i {
                out[i] += l[i][j] * z[j]
            }
        }
    }
    out
}

pub fn star_tracker_measurement(
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    l: Mat3,
    with_noise: bool,
    draw: Option<Vec3>,
) -> [f64; 4] {
    let mut q = normalize(mul(misalignment, mul(orientation, attitude)));
    if with_noise {
        let eta = mul3v(l, draw.unwrap_or_else(|| normals3(0, 0)));
        let dq = normalize([1., 0.5 * eta[0], 0.5 * eta[1], 0.5 * eta[2]]);
        q = normalize(mul(dq, q));
    }
    q
}
pub fn small_rotation_noise_quaternion_from_cholesky(l: Mat3, draw: Option<Vec3>) -> [f64; 4] {
    let eta = mul3v(l, draw.unwrap_or_else(|| normals3(0, 0)));
    normalize([1., 0.5 * eta[0], 0.5 * eta[1], 0.5 * eta[2]])
}
pub fn star_tracker_residual(measured: [f64; 4], estimated: [f64; 4]) -> Vec3 {
    let mut q = normalize(mul(measured, conj(estimated)));
    if q[0] < 0. {
        q = q.map(|x| -x)
    }
    [2. * q[1], 2. * q[2], 2. * q[3]]
}
pub fn star_tracker_mrp_jacobian(d: Vec3) -> Mat3 {
    let s = dot(d, d);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            4. / (1. + s).powi(2) * ((if i == j { 1. + s } else { 0. }) - 2. * d[i] * d[j])
        })
    })
}
pub fn star_tracker_information(cov: Mat3, orientation: [f64; 4], misalignment: [f64; 4]) -> Mat3 {
    let m = mounting(orientation, misalignment);
    matmul(transpose(m), solve3(cov, m))
}
pub fn magnetometer_information(
    field: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    noise: Mat3,
    field_cov: Mat3,
) -> Mat3 {
    let m = mounting(orientation, misalignment);
    let r = dcm(attitude);
    let f = mul3v(m, mul3v(r, field));
    let c = addm(
        noise,
        matmul(
            matmul(matmul(matmul(m, r), field_cov), transpose(r)),
            transpose(m),
        ),
    );
    let k = skew(f);
    matmul(
        matmul(matmul(matmul(transpose(m), transpose(k)), inverse3(c)), k),
        m,
    )
}
pub fn coarse_sun_available(
    sun: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    normal: Vec3,
    fov: f64,
    enabled: bool,
    visible: bool,
) -> bool {
    if !enabled || !visible {
        return false;
    }
    let sb = rotate(attitude, sun);
    let nb = mul3v(transpose(mounting(orientation, misalignment)), normal);
    dot(nb, sb) >= fov.cos()
}
pub fn coarse_sun_information(
    sun: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    normal: Vec3,
    sensitivity: f64,
    variance: f64,
) -> Mat3 {
    let sb = rotate(attitude, sun);
    let nb = mul3v(transpose(mounting(orientation, misalignment)), normal);
    let d = cross(nb, sb);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| sensitivity * sensitivity / variance * d[i] * d[j])
    })
}
pub fn fine_direction_available(
    direction: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
    fov: [f64; 2],
    enabled: bool,
    visible: bool,
) -> bool {
    if !enabled || !visible {
        return false;
    }
    let s = mul3v(
        mounting(orientation, misalignment),
        rotate(attitude, direction),
    );
    s[0].atan2(s[2]).abs() <= fov[0] && s[1].atan2(s[2]).abs() <= fov[1]
}

pub fn rotate(q: [f64; 4], v: Vec3) -> Vec3 {
    mul3v(dcm(q), v)
}
pub fn skew_matrix(v: Vec3) -> Mat3 {
    [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]]
}
pub fn attitude_vector_jacobian_quaternion(v: Vec3, delta: [f64; 4]) -> Mat3 {
    let q = normalize(delta);
    let r = rotate(q, v);
    let s = skew_matrix(r);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let right = [
        [w + x * x / w, -z + y / w * x, y + x / w * z],
        [z + x / w * y, w + y * y / w, -x + z / w * y],
        [-y + x / w * z, x + y / w * z, w + z * z / w],
    ];
    matmul(s, right).map(|row| row.map(|x| -2. * x))
}
pub fn attitude_vector_jacobian_mrp(v: Vec3, d: Vec3) -> Mat3 {
    let square = dot(d, d);
    let den = 1. + square;
    let q = [
        (1. - square) / den,
        2. * d[0] / den,
        2. * d[1] / den,
        2. * d[2] / den,
    ];
    let r = rotate(q, v);
    let s = skew_matrix(r);
    let p = [
        [
            1. - square + 2. * d[0] * d[0],
            2. * (d[0] * d[1] - d[2]),
            2. * (d[0] * d[2] + d[1]),
        ],
        [
            2. * (d[1] * d[0] + d[2]),
            1. - square + 2. * d[1] * d[1],
            2. * (d[1] * d[2] - d[0]),
        ],
        [
            2. * (d[2] * d[0] - d[1]),
            2. * (d[2] * d[1] + d[0]),
            1. - square + 2. * d[2] * d[2],
        ],
    ];
    matmul(s, p).map(|row| row.map(|x| -4. * x / den.powi(2)))
}
pub fn attitude_vector_jacobian_angle(v: Vec3, d: Vec3) -> Mat3 {
    let t = dot(d, d).sqrt();
    let q = if t < 1e-14 {
        [1., 0., 0., 0.]
    } else {
        let h = (t / 2.).sin() / t;
        [(t / 2.).cos(), h * d[0], h * d[1], h * d[2]]
    };
    let s = skew_matrix(rotate(q, v));
    let cross = skew_matrix(d);
    let cross2 = matmul(cross, cross);
    let a = if t < 1e-5 {
        0.5 - t * t / 24.
    } else {
        (1. - t.cos()) / (t * t)
    };
    let b = if t < 1e-5 {
        1. / 6. - t * t / 120.
    } else {
        (t - t.sin()) / (t * t * t)
    };
    let j = std::array::from_fn(|i| {
        std::array::from_fn(|k| (if i == k { 1. } else { 0. }) + a * cross[i][k] + b * cross2[i][k])
    });
    matmul(s, j).map(|row| row.map(|x| -x))
}
pub fn mounting(o: [f64; 4], m: [f64; 4]) -> Mat3 {
    dcm(mul(m, o))
}
pub fn rotate_inertial_to_sensor(
    v: Vec3,
    attitude: [f64; 4],
    orientation: [f64; 4],
    misalignment: [f64; 4],
) -> Vec3 {
    mul3v(mounting(orientation, misalignment), rotate(attitude, v))
}
pub fn rotate_body_to_sensor(v: Vec3, orientation: [f64; 4], misalignment: [f64; 4]) -> Vec3 {
    mul3v(mounting(orientation, misalignment), v)
}
pub fn dcm(q: [f64; 4]) -> Mat3 {
    let q = normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}
fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn add3(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|i| a[i] + b[i])
}
fn add2(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] + b[0], a[1] + b[1]]
}
fn mul2v(a: Mat2, v: Vec2) -> Vec2 {
    std::array::from_fn(|i| (0..2).map(|j| a[i][j] * v[j]).sum())
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn mul3v(a: Mat3, v: Vec3) -> Vec3 {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * v[j]).sum())
}
fn transpose(a: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn matmul(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn addm(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn skew(v: Vec3) -> Mat3 {
    [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]]
}
fn solve3(a: Mat3, b: Mat3) -> Mat3 {
    matmul(inverse3(a), b)
}
fn inverse3(a: Mat3) -> Mat3 {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    [
        [
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
        ],
        [
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
        ],
        [
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ],
    ]
    .map(|r| r.map(|x| x / det))
}
fn normals3(seed: u64, tick: u64) -> Vec3 {
    let v = philox_native::philox::standard_normals(seed, tick, 0, 3);
    [v[0], v[1], v[2]]
}
fn normals(seed: u64, tick: u64, n: usize) -> Vec<f64> {
    philox_native::philox::standard_normals(seed, tick, 0, n)
}
