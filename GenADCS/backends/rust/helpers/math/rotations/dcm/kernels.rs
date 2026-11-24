pub fn axis_angle_to_dcm_native(axis: [f64; 3], angle: f64) -> Result<M, &'static str> {
    let quaternion = axis_angle_to_quaternion_native(axis, angle)?;
    Ok(quaternion_to_dcm_native(quaternion))
}

pub fn rotate_vector_dcm_native(matrix: M, vector: [f64; 3]) -> [f64; 3] {
    mv(matrix, vector)
}

pub fn inverse_dcm_native(matrix: M) -> M {
    tr(matrix)
}

pub fn compose_dcm_native(left: M, right: M) -> M {
    mm(left, right)
}

pub fn relative_dcm_native(target: M, reference: M) -> M {
    mm(target, tr(reference))
}

pub fn dcm_kernel_orthogonality_error_native(matrix: M) -> f64 {
    (matrix.transpose() * matrix - M::identity()).norm()
}

pub fn normalize_dcm_native(matrix: M) -> Result<M, &'static str> {
    let mut first = matrix.column(0).into_owned();
    let norm_first = first.norm();
    if norm_first == 0.0 {
        return Err("DCM has a degenerate first column");
    }
    first /= norm_first;

    let second_original = matrix.column(1).into_owned();
    let mut second = second_original - first.dot(&second_original) * first;
    let norm_second = second.norm();
    if norm_second == 0.0 {
        return Err("DCM has degenerate first and second columns");
    }
    second /= norm_second;
    let third = first.cross(&second);
    Ok(M::from_columns(&[first, second, third]))
}
