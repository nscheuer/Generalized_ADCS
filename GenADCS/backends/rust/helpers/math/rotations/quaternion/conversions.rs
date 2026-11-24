pub fn axis_angle_to_quaternion_native(axis: [f64; 3], angle: f64) -> Result<[f64; 4], &'static str> {
    let norm = dot(&axis, &axis).sqrt();
    if norm == 0.0 {
        return Err("axis must be nonzero");
    }
    let scale = (angle / 2.0).sin() / norm;
    Ok([
        (angle / 2.0).cos(),
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
    ])
}

pub fn dcm_to_quaternion_native(m: M) -> Result<[f64; 4], &'static str> {
    let trace = m.trace();
    let q = if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        [
            0.25 * s,
            (m[(2, 1)] - m[(1, 2)]) / s,
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(1, 0)] - m[(0, 1)]) / s,
        ]
    } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
        let s = 2.0 * (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt();
        [
            (m[(2, 1)] - m[(1, 2)]) / s,
            0.25 * s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
        ]
    } else if m[(1, 1)] > m[(2, 2)] {
        let s = 2.0 * (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt();
        [
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            0.25 * s,
            (m[(1, 2)] + m[(2, 1)]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt();
        [
            (m[(1, 0)] - m[(0, 1)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
            (m[(1, 2)] + m[(2, 1)]) / s,
            0.25 * s,
        ]
    };
    let q = qnormalize_native(q)?;
    Ok(if q[0] < 0.0 { q.map(|v| -v) } else { q })
}
