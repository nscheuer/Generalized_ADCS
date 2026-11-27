pub const ALLOC_L2: i32 = 0;
pub const ALLOC_LINFINITY: i32 = 1;

pub fn saturate_wheels(mut u_rw: Vec<f64>, max_torque: &[f64]) -> Vec<f64> {
    for (u, &lim) in u_rw.iter_mut().zip(max_torque.iter()) {
        if *u > lim {
            *u = lim;
        } else if *u < -lim {
            *u = -lim;
        }
    }
    u_rw
}

pub trait Allocator: Send + Sync {
    fn kind(&self) -> i32;
}
