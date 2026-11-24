pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    (0..N).map(|i| a[i] * b[i]).sum()
}
pub fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
    ]
}
pub fn qnorm(q: [f64; 4]) -> f64 {
    dot(&q, &q).sqrt()
}
pub fn qnormalize_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    let norm = qnorm(q);
    if norm == 0.0 {
        Err("quaternion must have nonzero norm")
    } else {
        Ok(q.map(|value| value / norm))
    }
}
pub fn qconj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
pub fn qinv_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    let norm_squared = dot(&q, &q);
    if norm_squared == 0.0 {
        Err("zero quaternion has no inverse")
    } else {
        Ok(qconj(q).map(|value| value / norm_squared))
    }
}
pub fn qrotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let c = cross3(u, v);
    let cc = cross3(u, c);
    [
        v[0] + 2.0 * (q[0] * c[0] + cc[0]),
        v[1] + 2.0 * (q[0] * c[1] + cc[1]),
        v[2] + 2.0 * (q[0] * c[2] + cc[2]),
    ]
}
pub fn qdcm(q: [f64; 4]) -> M {
    let [w, x, y, z] = q;
    M::from_row_slice(&[
        w * w + x * x - y * y - z * z,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    ])
}
pub fn mm(a: M, b: M) -> M {
    a * b
}
pub fn tr(a: M) -> M {
    a.transpose()
}
pub fn mv(a: M, b: [f64; 3]) -> [f64; 3] {
    let v = a * SVector::<f64, 3>::from_row_slice(&b);
    [v[0], v[1], v[2]]
}
pub fn proper_dcm(m: M) -> bool {
    (m.transpose() * m - M::identity()).norm() <= 1e-8 && (m.determinant() - 1.0).abs() <= 1e-8
}
