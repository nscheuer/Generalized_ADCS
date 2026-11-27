use nalgebra::{DMatrix, SymmetricEigen};

use crate::controllers::allocator::{
    allocate_l2, allocate_linfinity, Allocator, LInfinityAllocator,
};

fn null_basis(a_mat: &DMatrix<f64>) -> DMatrix<f64> {
    let n_w = a_mat.ncols();
    let eig = SymmetricEigen::new(a_mat.transpose() * a_mat);
    let max_lam = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let lam_tol = 1e-12 * max_lam.max(1.0);
    let cols: Vec<usize> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, &lam)| lam <= lam_tol)
        .map(|(i, _)| i)
        .collect();
    let mut nb = DMatrix::zeros(n_w, cols.len());
    for (c_out, &c_in) in cols.iter().enumerate() {
        for r in 0..n_w {
            nb[(r, c_out)] = eig.eigenvectors[(r, c_in)];
        }
    }
    nb
}

#[test]
fn test_linfinity_allocator_minimizes_max_wheel_torque_on_4_wheels() {
    let s3 = 1.0 / 3.0_f64.sqrt();
    let a_mat = DMatrix::from_row_slice(
        3,
        4,
        &[1.0, 0.0, 0.0, s3, 0.0, 1.0, 0.0, s3, 0.0, 0.0, 1.0, s3],
    );
    let a_pinv = a_mat.clone().pseudo_inverse(1e-15).unwrap();
    let null_proj = DMatrix::identity(4, 4) - &a_pinv * &a_mat;
    let nb = null_basis(&a_mat);
    let max_t = [1.0; 4];
    let tau_c = [0.3, 0.0, 0.0];

    let u_l2 = allocate_l2(tau_c, &a_pinv, &null_proj, &[0.0; 4], &max_t);
    let u_linf = allocate_linfinity(tau_c, &a_pinv, &nb, &null_proj, &[0.0; 4], &max_t);
    assert_eq!(LInfinityAllocator.kind(), 1);
    let prod = &a_mat * DMatrix::from_column_slice(4, 1, &u_linf);
    for i in 0..3 {
        assert!((prod[(i, 0)] - tau_c[i]).abs() < 1e-9);
    }
    let max_l2 = u_l2.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_linf = u_linf.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(max_linf < max_l2);
}

#[test]
fn test_linfinity_allocator_general_5_wheel_nullspace() {
    let a_mat = DMatrix::from_row_slice(
        3,
        5,
        &[
            1.0, 0.0, 0.0, 0.5, -0.5,
            0.0, 1.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 1.0, 0.70710678, 0.70710678,
        ],
    );
    let a_pinv = a_mat.clone().pseudo_inverse(1e-15).unwrap();
    let null_proj = DMatrix::identity(5, 5) - &a_pinv * &a_mat;
    let nb = null_basis(&a_mat);
    let tau_c = [0.4, 0.1, -0.1];
    let max_t = [1.0; 5];

    let u_l2 = allocate_l2(tau_c, &a_pinv, &null_proj, &[0.0; 5], &max_t);
    let u_linf = allocate_linfinity(tau_c, &a_pinv, &nb, &null_proj, &[0.0; 5], &max_t);
    let prod = &a_mat * DMatrix::from_column_slice(5, 1, &u_linf);
    for i in 0..3 {
        assert!((prod[(i, 0)] - tau_c[i]).abs() < 1e-8);
    }
    let max_l2 = u_l2.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_linf = u_linf.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(max_linf <= max_l2 + 1e-9);
}
