pub fn mrp_normalize_native(mut p: [f64; 3]) -> [f64; 3] {
    let norm_squared = dot(&p, &p);
    if norm_squared > 1.0 {
        p = p.map(|value| -value / norm_squared);
    }
    p
}

pub fn mrp_norm_native(p: [f64; 3]) -> f64 {
    dot(&p, &p).sqrt()
}

pub fn mrp_switch_shadow_set_native(p: [f64; 3]) -> Result<[f64; 3], &'static str> {
    let norm_squared = dot(&p, &p);
    if norm_squared == 0.0 {
        return Err("the identity MRP has no finite shadow set");
    }
    Ok(p.map(|value| -value / norm_squared))
}

pub fn mrp_conjugate_native(p: [f64; 3]) -> [f64; 3] {
    let q = mrp_to_quaternion_native(p);
    quaternion_to_mrp_native(quaternion_conjugate_native(q)).unwrap_or(p)
}

pub fn mrp_multiply_native(p: [f64; 3], other: [f64; 3]) -> Result<[f64; 3], &'static str> {
    quaternion_to_mrp_native(quaternion_multiply_native(
        mrp_to_quaternion_native(p),
        mrp_to_quaternion_native(other),
    ))
}

pub fn mrp_relative_native(p: [f64; 3], other: [f64; 3]) -> Result<[f64; 3], &'static str> {
    let q = quaternion_relative_native(
        mrp_to_quaternion_native(p),
        mrp_to_quaternion_native(other),
    )?;
    quaternion_to_mrp_native(q)
}
