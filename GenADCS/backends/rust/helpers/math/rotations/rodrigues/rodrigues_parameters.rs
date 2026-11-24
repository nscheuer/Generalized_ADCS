pub fn rodrigues_norm_native(parameters: [f64; 3]) -> f64 {
    dot(&parameters, &parameters).sqrt()
}

pub fn rodrigues_from_quaternion_native(quaternion: [f64; 4]) -> Result<[f64; 3], &'static str> {
    quaternion_to_rodrigues_native(quaternion)
}

pub fn rodrigues_to_quaternion_parameters_native(parameters: [f64; 3]) -> [f64; 4] {
    rodrigues_to_quaternion_native(parameters)
}

pub fn rodrigues_rotate_native(parameters: [f64; 3], vector: [f64; 3]) -> [f64; 3] {
    quaternion_rotate_native(rodrigues_to_quaternion_native(parameters), vector)
}

pub fn rodrigues_multiply_native(
    parameters: [f64; 3],
    other: [f64; 3],
) -> Result<[f64; 3], &'static str> {
    quaternion_to_rodrigues_native(quaternion_multiply_native(
        rodrigues_to_quaternion_native(parameters),
        rodrigues_to_quaternion_native(other),
    ))
}

pub fn rodrigues_relative_native(
    parameters: [f64; 3],
    reference: [f64; 3],
) -> Result<[f64; 3], &'static str> {
    let q = quaternion_relative_native(
        rodrigues_to_quaternion_native(parameters),
        rodrigues_to_quaternion_native(reference),
    )?;
    quaternion_to_rodrigues_native(q)
}
