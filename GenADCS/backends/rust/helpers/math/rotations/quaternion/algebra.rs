pub fn normalize_quaternion_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    qnormalize_native(q)
}

pub fn quaternion_norm_native(q: [f64; 4]) -> f64 {
    qnorm(q)
}

pub fn jpl_to_hamilton_native([x, y, z, w]: [f64; 4]) -> [f64; 4] {
    [w, -x, -y, -z]
}

pub fn hamilton_to_jpl_native([w, x, y, z]: [f64; 4]) -> [f64; 4] {
    [-x, -y, -z, w]
}

pub fn conjugate_native(q: [f64; 4]) -> [f64; 4] {
    qconj(q)
}

pub fn inverse_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    qinv_native(q)
}

pub fn multiply_native(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    qmul(a, b)
}

pub fn relative_quaternion_native(r: [f64; 4], t: [f64; 4]) -> Result<[f64; 4], &'static str> {
    let q = qnormalize_native(qmul(t, qinv_native(r)?))?;
    Ok(if q[0] < 0.0 { q.map(|value| -value) } else { q })
}

pub fn rotate_vector_native(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    qrotate(q, v)
}

pub fn quaternion_to_dcm_native(q: [f64; 4]) -> M {
    qdcm(q)
}
