pub fn rotation_vector_to_quaternion_native(v: [f64; 3]) -> Result<[f64; 4], &'static str> {
    let angle = dot(&v, &v).sqrt();
    if angle < 1e-12 {
        qnormalize_native([
            1.0 - angle * angle / 8.0,
            v[0] * 0.5,
            v[1] * 0.5,
            v[2] * 0.5,
        ])
    } else {
        let scale = (angle / 2.0).sin() / angle;
        Ok([
            (angle / 2.0).cos(),
            v[0] * scale,
            v[1] * scale,
            v[2] * scale,
        ])
    }
}

pub fn quaternion_to_rotation_vector_native(q: [f64; 4]) -> Result<[f64; 3], &'static str> {
    let mut q = qnormalize_native(q)?;
    if q[0] < 0.0 {
        q = q.map(|value| -value);
    }
    let vector_norm = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if vector_norm < 1e-12 {
        Ok([2.0 * q[1], 2.0 * q[2], 2.0 * q[3]])
    } else {
        let scale = 2.0 * vector_norm.atan2(q[0]) / vector_norm;
        Ok([q[1] * scale, q[2] * scale, q[3] * scale])
    }
}

pub fn slerp_native(a: [f64; 4], mut b: [f64; 4], t: f64) -> Result<[f64; 4], &'static str> {
    let a = qnormalize_native(a)?;
    b = qnormalize_native(b)?;
    let mut dot_ab = dot(&a, &b);
    if dot_ab < 0.0 {
        b = b.map(|value| -value);
        dot_ab = -dot_ab;
    }
    if dot_ab > 0.9995 {
        qnormalize_native(std::array::from_fn(|i| a[i] + t * (b[i] - a[i])))
    } else {
        let angle = dot_ab.clamp(-1.0, 1.0).acos();
        let denominator = angle.sin();
        let weight_a = ((1.0 - t) * angle).sin() / denominator;
        let weight_b = (t * angle).sin() / denominator;
        Ok(std::array::from_fn(|i| weight_a * a[i] + weight_b * b[i]))
    }
}
