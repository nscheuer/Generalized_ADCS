pub fn mrp_to_quaternion_native(parameters: [f64; 3]) -> [f64; 4] {
    let norm_squared = dot(&parameters, &parameters);
    let denominator = 1.0 + norm_squared;
    [
        (1.0 - norm_squared) / denominator,
        2.0 * parameters[0] / denominator,
        2.0 * parameters[1] / denominator,
        2.0 * parameters[2] / denominator,
    ]
}

pub fn quaternion_to_mrp_native(quaternion: [f64; 4]) -> Result<[f64; 3], &'static str> {
    let mut q = qnormalize_native(quaternion)?;
    if q[0] < 0.0 {
        q = q.map(|value| -value);
    }
    let mut parameters = [
        q[1] / (1.0 + q[0]),
        q[2] / (1.0 + q[0]),
        q[3] / (1.0 + q[0]),
    ];
    let norm_squared = dot(&parameters, &parameters);
    if norm_squared > 1.0 {
        parameters = parameters.map(|value| -value / norm_squared);
    }
    Ok(parameters)
}

pub fn mrp_shadow_set_native(parameters: [f64; 3]) -> Result<[f64; 3], &'static str> {
    let norm_squared = dot(&parameters, &parameters);
    if norm_squared == 0.0 {
        return Err("the identity MRP has no finite shadow set");
    }
    Ok(parameters.map(|value| -value / norm_squared))
}
