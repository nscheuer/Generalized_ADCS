use math_native::quaternion_to_mrp_native;

use super::feedback::{to_gain_matrix, Feedback, GainInput, Mat3, FB_MRP_FEEDBACK_LINEARIZED};

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mat3_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_t_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

fn mrp_b_matrix(sigma: [f64; 3]) -> Mat3 {
    let s2 = dot3(sigma, sigma);
    let d = 1.0 - s2;
    [
        [
            d + 2.0 * sigma[0] * sigma[0],
            -2.0 * sigma[2] + 2.0 * sigma[0] * sigma[1],
            2.0 * sigma[1] + 2.0 * sigma[0] * sigma[2],
        ],
        [
            2.0 * sigma[2] + 2.0 * sigma[1] * sigma[0],
            d + 2.0 * sigma[1] * sigma[1],
            -2.0 * sigma[0] + 2.0 * sigma[1] * sigma[2],
        ],
        [
            -2.0 * sigma[1] + 2.0 * sigma[2] * sigma[0],
            2.0 * sigma[0] + 2.0 * sigma[2] * sigma[1],
            d + 2.0 * sigma[2] * sigma[2],
        ],
    ]
}

fn mrp_b_dot_matrix(sigma: [f64; 3], sigma_dot: [f64; 3]) -> Mat3 {
    let d = -2.0 * dot3(sigma, sigma_dot);
    [
        [
            d + 4.0 * sigma_dot[0] * sigma[0],
            -2.0 * sigma_dot[2] + 2.0 * (sigma_dot[0] * sigma[1] + sigma[0] * sigma_dot[1]),
            2.0 * sigma_dot[1] + 2.0 * (sigma_dot[0] * sigma[2] + sigma[0] * sigma_dot[2]),
        ],
        [
            2.0 * sigma_dot[2] + 2.0 * (sigma_dot[1] * sigma[0] + sigma[1] * sigma_dot[0]),
            d + 4.0 * sigma_dot[1] * sigma[1],
            -2.0 * sigma_dot[0] + 2.0 * (sigma_dot[1] * sigma[2] + sigma[1] * sigma_dot[2]),
        ],
        [
            -2.0 * sigma_dot[1] + 2.0 * (sigma_dot[2] * sigma[0] + sigma[2] * sigma_dot[0]),
            2.0 * sigma_dot[0] + 2.0 * (sigma_dot[2] * sigma[1] + sigma[2] * sigma_dot[1]),
            d + 4.0 * sigma_dot[2] * sigma[2],
        ],
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn mrp_feedback_linearized_step(
    q: [f64; 4],
    omega: [f64; 3],
    q_d: [f64; 4],
    omega_d: [f64; 3],
    omega_d_dot: [f64; 3],
    ff_torque: [f64; 3],
    wheel_momenta: &[f64],
    total_inertia: &Mat3,
    bus_inertia: &Mat3,
    wheel_axes: &[[f64; 3]],
    kp: &Mat3,
    kd: &Mat3,
) -> [f64; 3] {
    let mrp = quaternion_to_mrp_native(q).unwrap_or([0.0; 3]);
    let sigma = [-mrp[0], -mrp[1], -mrp[2]];

    let mrp_d = quaternion_to_mrp_native(q_d).unwrap_or([0.0; 3]);
    let mut sigma_d = [-mrp_d[0], -mrp_d[1], -mrp_d[2]];

    let sd2 = dot3(sigma_d, sigma_d);
    if sd2 > 1e-12 {
        let sigma_d_shadow = [-sigma_d[0] / sd2, -sigma_d[1] / sd2, -sigma_d[2] / sd2];
        let diff_direct = [
            sigma[0] - sigma_d[0],
            sigma[1] - sigma_d[1],
            sigma[2] - sigma_d[2],
        ];
        let diff_shadow = [
            sigma[0] - sigma_d_shadow[0],
            sigma[1] - sigma_d_shadow[1],
            sigma[2] - sigma_d_shadow[2],
        ];
        if dot3(diff_shadow, diff_shadow) < dot3(diff_direct, diff_direct) {
            sigma_d = sigma_d_shadow;
        }
    }

    let b = mrp_b_matrix(sigma);
    let b_omega = mat3_vec(&b, omega);
    let sigma_dot = [0.25 * b_omega[0], 0.25 * b_omega[1], 0.25 * b_omega[2]];
    let b_dot = mrp_b_dot_matrix(sigma, sigma_dot);

    let b_d = mrp_b_matrix(sigma_d);
    let bd_wd = mat3_vec(&b_d, omega_d);
    let sigma_d_dot = [0.25 * bd_wd[0], 0.25 * bd_wd[1], 0.25 * bd_wd[2]];
    let b_d_dot = mrp_b_dot_matrix(sigma_d, sigma_d_dot);
    let bdd_wd = mat3_vec(&b_d_dot, omega_d);
    let bd_wdd = mat3_vec(&b_d, omega_d_dot);
    let sigma_d_ddot = [
        0.25 * bdd_wd[0] + 0.25 * bd_wdd[0],
        0.25 * bdd_wd[1] + 0.25 * bd_wdd[1],
        0.25 * bdd_wd[2] + 0.25 * bd_wdd[2],
    ];

    let e = [
        sigma[0] - sigma_d[0],
        sigma[1] - sigma_d[1],
        sigma[2] - sigma_d[2],
    ];
    let e_dot = [
        sigma_dot[0] - sigma_d_dot[0],
        sigma_dot[1] - sigma_d_dot[1],
        sigma_dot[2] - sigma_d_dot[2],
    ];

    let b_dot_w = mat3_vec(&b_dot, omega);
    let kd_ed = mat3_vec(kd, e_dot);
    let kp_e = mat3_vec(kp, e);
    let rhs = [
        -0.25 * b_dot_w[0] + sigma_d_ddot[0] - kd_ed[0] - kp_e[0],
        -0.25 * b_dot_w[1] + sigma_d_ddot[1] - kd_ed[1] - kp_e[1],
        -0.25 * b_dot_w[2] + sigma_d_ddot[2] - kd_ed[2] - kp_e[2],
    ];

    let s2 = dot3(sigma, sigma);
    let scale = 4.0 / ((1.0 + s2) * (1.0 + s2));
    let bt_rhs = mat3_t_vec(&b, rhs);
    let nu = [scale * bt_rhs[0], scale * bt_rhs[1], scale * bt_rhs[2]];

    let mut angular_momentum = mat3_vec(total_inertia, omega);
    for (axis, &h) in wheel_axes.iter().zip(wheel_momenta.iter()) {
        angular_momentum[0] += axis[0] * h;
        angular_momentum[1] += axis[1] * h;
        angular_momentum[2] += axis[2] * h;
    }

    let gyro = cross3(omega, angular_momentum);
    let j_nu = mat3_vec(bus_inertia, nu);
    [
        gyro[0] + j_nu[0] + ff_torque[0],
        gyro[1] + j_nu[1] + ff_torque[1],
        gyro[2] + j_nu[2] + ff_torque[2],
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct MRPFeedbackLinearized {
    pub kp: Mat3,
    pub kd: Mat3,
}

impl MRPFeedbackLinearized {
    pub fn new(kp: impl Into<GainInput>, kd: impl Into<GainInput>) -> Result<Self, String> {
        let kp_mat = to_gain_matrix(kp.into(), "kp", true)?;
        let kd_mat = to_gain_matrix(kd.into(), "kd", false)?;
        Ok(Self {
            kp: kp_mat,
            kd: kd_mat,
        })
    }
}

impl Feedback for MRPFeedbackLinearized {
    fn kind(&self) -> i32 {
        FB_MRP_FEEDBACK_LINEARIZED
    }

    fn to_matrices(&self) -> (Mat3, Mat3) {
        (self.kp, self.kd)
    }
}
