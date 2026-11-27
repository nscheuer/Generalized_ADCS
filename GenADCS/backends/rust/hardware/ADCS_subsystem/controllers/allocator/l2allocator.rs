use nalgebra::DMatrix;

use super::allocator::{saturate_wheels, Allocator, ALLOC_L2};

pub fn allocate_l2(
    tau_c: [f64; 3],
    a_pinv: &DMatrix<f64>,
    null_proj: &DMatrix<f64>,
    u_0: &[f64],
    max_torque: &[f64],
) -> Vec<f64> {
    let n_w = a_pinv.nrows();
    let mut u_rw = vec![0.0; n_w];
    for i in 0..n_w {
        let mut val = a_pinv[(i, 0)] * tau_c[0] + a_pinv[(i, 1)] * tau_c[1] + a_pinv[(i, 2)] * tau_c[2];
        for j in 0..u_0.len() {
            val += null_proj[(i, j)] * u_0[j];
        }
        u_rw[i] = val;
    }
    saturate_wheels(u_rw, max_torque)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct L2Allocator;

impl Allocator for L2Allocator {
    fn kind(&self) -> i32 {
        ALLOC_L2
    }
}
