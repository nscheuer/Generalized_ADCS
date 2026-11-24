pub fn rodrigues_to_quaternion_native(parameters: [f64; 3]) -> [f64; 4] {
    let denominator = (1.0 + dot(&parameters, &parameters)).sqrt();
    [
        1.0 / denominator,
        parameters[0] / denominator,
        parameters[1] / denominator,
        parameters[2] / denominator,
    ]
}

pub fn quaternion_to_rodrigues_native(
    quaternion: [f64; 4],
) -> Result<[f64; 3], &'static str> {
    let mut q = qnormalize_native(quaternion)?;
    if q[0] < 0.0 {
        q = q.map(|value| -value);
    }
    if q[0].abs() <= 1e-12 {
        return Err("Rodrigues parameters are singular at a 180-degree rotation");
    }
    Ok([q[1] / q[0], q[2] / q[0], q[3] / q[0]])
}
