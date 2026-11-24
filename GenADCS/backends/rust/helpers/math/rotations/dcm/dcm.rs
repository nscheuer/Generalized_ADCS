pub fn dcm_identity_native() -> M { M::identity() }

pub fn dcm_is_proper_native(matrix: M) -> bool { proper_dcm(matrix) }

pub fn dcm_inverse_native(matrix: M) -> M { tr(matrix) }

pub fn dcm_multiply_native(left: M, right: M) -> M { mm(left, right) }

pub fn dcm_rotate_native(matrix: M, vector: [f64; 3]) -> [f64; 3] { mv(matrix, vector) }

pub fn dcm_relative_native(target: M, reference: M) -> M {
    mm(target, tr(reference))
}

pub fn dcm_orthogonality_error_native(matrix: M) -> f64 {
    (tr(matrix) * matrix - M::identity()).norm()
}
