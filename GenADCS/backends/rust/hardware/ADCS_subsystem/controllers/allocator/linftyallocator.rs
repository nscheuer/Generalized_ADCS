use nalgebra::DMatrix;

use super::allocator::{saturate_wheels, Allocator, ALLOC_LINFINITY};

pub fn allocate_linfinity(
    tau_c: [f64; 3],
    a_pinv: &DMatrix<f64>,
    null_basis: &DMatrix<f64>,
    null_proj: &DMatrix<f64>,
    u_0: &[f64],
    max_torque: &[f64],
) -> Vec<f64> {
    let n_w = a_pinv.nrows();
    let mut u_base = vec![0.0; n_w];
    for i in 0..n_w {
        u_base[i] =
            a_pinv[(i, 0)] * tau_c[0] + a_pinv[(i, 1)] * tau_c[1] + a_pinv[(i, 2)] * tau_c[2];
    }

    let r = null_basis.ncols();
    if r == 0 {
        return saturate_wheels(u_base, max_torque);
    }

    let u = if r == 1 {
        let mut best_u = u_base.clone();
        let mut best_max = best_u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for i in 0..n_w {
            for j in (i + 1)..n_w {
                for sgn in [1.0, -1.0] {
                    let denom = null_basis[(i, 0)] - sgn * null_basis[(j, 0)];
                    if denom.abs() > 1e-12 {
                        let z = -(u_base[i] - sgn * u_base[j]) / denom;
                        let mut cand_max = 0.0_f64;
                        for k in 0..n_w {
                            let a = (u_base[k] + z * null_basis[(k, 0)]).abs();
                            if a > cand_max {
                                cand_max = a;
                            }
                        }
                        if cand_max < best_max {
                            best_max = cand_max;
                            for k in 0..n_w {
                                best_u[k] = u_base[k] + z * null_basis[(k, 0)];
                            }
                        }
                    }
                }
            }
        }
        best_u
    } else {
        let mut z = vec![0.0; r];
        let mut u = u_base.clone();
        let mut best_u = u_base.clone();
        let mut best_max = best_u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for it in 0..40 {
            let mut k_max = 0;
            let mut u_max = u[0].abs();
            for (i, &val) in u.iter().enumerate().skip(1) {
                let a = val.abs();
                if a > u_max {
                    u_max = a;
                    k_max = i;
                }
            }
            if u_max < best_max {
                best_max = u_max;
                best_u.copy_from_slice(&u);
            }
            let sgn = if u[k_max] >= 0.0 { 1.0 } else { -1.0 };
            let mut g = vec![0.0; r];
            let mut g_sq = 0.0;
            for c in 0..r {
                g[c] = sgn * null_basis[(k_max, c)];
                g_sq += g[c] * g[c];
            }
            let g_norm = g_sq.sqrt();
            if g_norm <= 1e-12 {
                break;
            }
            let step = (0.4 * best_max / ((it + 1) as f64).sqrt()) / g_norm;
            for c in 0..r {
                z[c] -= step * g[c];
            }
            for i in 0..n_w {
                let mut val = u_base[i];
                for c in 0..r {
                    val += null_basis[(i, c)] * z[c];
                }
                u[i] = val;
            }
        }
        best_u
    };

    let mut u_rw = vec![0.0; n_w];
    for i in 0..n_w {
        let mut val = u[i];
        for j in 0..u_0.len() {
            val += null_proj[(i, j)] * u_0[j];
        }
        u_rw[i] = val;
    }
    saturate_wheels(u_rw, max_torque)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LInfinityAllocator;

impl Allocator for LInfinityAllocator {
    fn kind(&self) -> i32 {
        ALLOC_LINFINITY
    }
}
