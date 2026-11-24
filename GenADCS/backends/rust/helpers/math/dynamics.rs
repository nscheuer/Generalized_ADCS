pub fn wheel_momentum_derivative_native(
    torque: &[f64],
    alpha: [f64; 3],
    axes: &[[f64; 3]],
) -> Result<Vec<f64>, &'static str> {
    if torque.len() != axes.len() {
        return Err("wheel_axes row count must match wheel_torque size");
    }
    Ok(torque
        .iter()
        .zip(axes)
        .map(|(tau, axis)| tau - dot(axis, &alpha))
        .collect())
}

pub fn solve3_native(inertia: M, rhs: [f64; 3]) -> Result<[f64; 3], &'static str> {
    let rhs = SVector::<f64, 3>::from_row_slice(&rhs);
    let solution = inertia
        .full_piv_lu()
        .solve(&rhs)
        .ok_or("inertia matrix must be invertible")?;
    Ok([solution[0], solution[1], solution[2]])
}

pub fn rigid_body_angular_acceleration_native(
    w: [f64; 3],
    torque: [f64; 3],
    inertia: M,
) -> Result<[f64; 3], &'static str> {
    let omega = SVector::<f64, 3>::from_row_slice(&w);
    let torque = SVector::<f64, 3>::from_row_slice(&torque);
    let rhs = omega.cross(&(inertia * omega));
    solve3_native(inertia, (torque - rhs).into())
}

pub fn wheel_coupled_angular_acceleration_native(
    w: [f64; 3],
    h: &[f64],
    axes: &[[f64; 3]],
    torque: [f64; 3],
    inertia: M,
) -> Result<[f64; 3], &'static str> {
    if h.len() != axes.len() {
        return Err("wheel_axes row count must match wheel_momentum size");
    }
    let omega = SVector::<f64, 3>::from_row_slice(&w);
    let mut total = inertia * omega;
    for (momentum, axis) in h.iter().zip(axes) {
        total += SVector::<f64, 3>::from_row_slice(axis) * *momentum;
    }
    let torque = SVector::<f64, 3>::from_row_slice(&torque);
    solve3_native(inertia, (torque - omega.cross(&total)).into())
}
