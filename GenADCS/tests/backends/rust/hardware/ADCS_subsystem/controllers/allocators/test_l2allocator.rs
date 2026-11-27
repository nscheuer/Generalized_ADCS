use nalgebra::DMatrix;

use crate::controllers::allocator::{allocate_l2, Allocator, L2Allocator};

#[test]
fn test_l2_allocator_solves_minimum_norm_and_saturates() {
    let s3 = 1.0 / 3.0_f64.sqrt();
    let a_mat = DMatrix::from_row_slice(
        3,
        4,
        &[1.0, 0.0, 0.0, s3, 0.0, 1.0, 0.0, s3, 0.0, 0.0, 1.0, s3],
    );
    let a_pinv = a_mat.clone().pseudo_inverse(1e-15).unwrap();
    let null_proj = DMatrix::identity(4, 4) - &a_pinv * &a_mat;
    let tau_c = [0.3, -0.1, 0.2];
    let u = allocate_l2(tau_c, &a_pinv, &null_proj, &[0.0; 4], &[1.0; 4]);
    assert_eq!(L2Allocator.kind(), 0);
    let u_vec = DMatrix::from_column_slice(4, 1, &u);
    let prod = &a_mat * &u_vec;
    for i in 0..3 {
        assert!((prod[(i, 0)] - tau_c[i]).abs() < 1e-12);
    }

    let u_sat = allocate_l2([2.0, 0.0, 0.0], &a_pinv, &null_proj, &[0.0; 4], &[0.2; 4]);
    assert!(u_sat.iter().all(|&v| v.abs() <= 0.2 + 1e-12));
}
