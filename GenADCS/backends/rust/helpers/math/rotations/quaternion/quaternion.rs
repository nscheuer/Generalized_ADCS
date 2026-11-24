pub fn quaternion_hamilton_from_jpl_native([x, y, z, w]: [f64; 4]) -> [f64; 4] {
    [w, -x, -y, -z]
}

pub fn quaternion_jpl_from_hamilton_native([w, x, y, z]: [f64; 4]) -> [f64; 4] {
    [-x, -y, -z, w]
}

pub fn quaternion_normalize_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    qnormalize_native(q)
}

pub fn quaternion_conjugate_native(q: [f64; 4]) -> [f64; 4] { qconj(q) }

pub fn quaternion_inverse_native(q: [f64; 4]) -> Result<[f64; 4], &'static str> {
    qinv_native(q)
}

pub fn quaternion_multiply_native(a: [f64; 4], b: [f64; 4]) -> [f64; 4] { qmul(a, b) }

pub fn quaternion_rotate_native(q: [f64; 4], v: [f64; 3]) -> [f64; 3] { qrotate(q, v) }

pub fn quaternion_as_dcm_native(q: [f64; 4]) -> M { qdcm(q) }

pub fn quaternion_relative_native(q: [f64; 4], reference: [f64; 4]) -> Result<[f64; 4], &'static str> {
    let mut result = qnormalize_native(qmul(q, qinv_native(reference)?))?;
    if result[0] < 0.0 { result = result.map(|value| -value); }
    Ok(result)
}
