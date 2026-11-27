use std::sync::Arc;

use nalgebra::DMatrix;

use super::allocator::{allocate_l2, allocate_linfinity, Allocator, L2Allocator, ALLOC_L2};
use super::controller::{Controller, SensorConfig, CTRL_ATTITUDE};
use super::desaturation::{
    hogan_schaub_mtq_dump, hogan_schaub_u0, Desaturation, NoDesaturation, DESAT_HOGAN_SCHAUB,
};
use super::feedback::{mrp_feedback_linearized_step, Feedback, Mat3, FB_MRP_FEEDBACK_LINEARIZED};
use super::feedforward::{passthrough_step, Feedforward, Passthrough, FF_PASSTHROUGH};

#[allow(clippy::too_many_arguments)]
pub fn step(
    q: [f64; 4],
    omega: [f64; 3],
    q_goal: [f64; 4],
    w_goal: [f64; 3],
    wheel_momenta: &[f64],
    total_inertia: &Mat3,
    bus_inertia: &Mat3,
    wheel_axes: &[[f64; 3]],
    wheel_max_torque: &[f64],
    a_pinv: &DMatrix<f64>,
    null_basis: &DMatrix<f64>,
    null_proj: &DMatrix<f64>,
    ff_kind: i32,
    fb_kind: i32,
    alloc_kind: i32,
    desat_kind: i32,
    kp: &Mat3,
    kd: &Mat3,
    desat_gain: f64,
    desat_target: f64,
    desat_mtq_gain: f64,
    readings: &[f64],
    sensor_config: &SensorConfig,
    mtq_axes: &[[f64; 3]],
    mtq_max_dipole: &[f64],
    wheel_commands: &mut [f64],
    dipole_commands: &mut [f64],
) -> ([f64; 3], [f64; 4], [f64; 3], [f64; 3]) {
    let (q_d, w_d, w_d_dot, ff_torque) = if ff_kind == FF_PASSTHROUGH {
        passthrough_step(q_goal, w_goal, q)
    } else {
        (q, [0.0; 3], [0.0; 3], [0.0; 3])
    };

    let tau_c = if fb_kind == FB_MRP_FEEDBACK_LINEARIZED {
        mrp_feedback_linearized_step(
            q,
            omega,
            q_d,
            w_d,
            w_d_dot,
            ff_torque,
            wheel_momenta,
            total_inertia,
            bus_inertia,
            wheel_axes,
            kp,
            kd,
        )
    } else {
        ff_torque
    };

    if wheel_commands.is_empty() || !wheel_max_torque.iter().any(|&v| v > 0.0) {
        wheel_commands.fill(0.0);
        dipole_commands.fill(0.0);
        return (ff_torque, q_d, w_d, tau_c);
    }

    let u_0 = if desat_kind == DESAT_HOGAN_SCHAUB {
        hogan_schaub_mtq_dump(
            wheel_momenta,
            wheel_axes,
            desat_mtq_gain,
            desat_target,
            readings,
            sensor_config,
            mtq_axes,
            mtq_max_dipole,
            dipole_commands,
        );
        hogan_schaub_u0(wheel_momenta, desat_gain, desat_target)
    } else {
        dipole_commands.fill(0.0);
        vec![0.0; wheel_commands.len()]
    };

    let u_rw = if alloc_kind == ALLOC_L2 {
        allocate_l2(tau_c, a_pinv, null_proj, &u_0, wheel_max_torque)
    } else {
        allocate_linfinity(
            tau_c,
            a_pinv,
            null_basis,
            null_proj,
            &u_0,
            wheel_max_torque,
        )
    };
    wheel_commands.copy_from_slice(&u_rw);

    (ff_torque, q_d, w_d, tau_c)
}

#[derive(Clone)]
pub struct AttitudeController {
    pub feedback: Arc<dyn Feedback>,
    pub feedforward: Arc<dyn Feedforward>,
    pub allocator: Arc<dyn Allocator>,
    pub desaturation: Arc<dyn Desaturation>,
}

impl AttitudeController {
    pub fn new(
        feedback: Arc<dyn Feedback>,
        feedforward: Option<Arc<dyn Feedforward>>,
        allocator: Option<Arc<dyn Allocator>>,
        desaturation: Option<Arc<dyn Desaturation>>,
    ) -> Self {
        Self {
            feedback,
            feedforward: feedforward.unwrap_or_else(|| Arc::new(Passthrough)),
            allocator: allocator.unwrap_or_else(|| Arc::new(L2Allocator)),
            desaturation: desaturation.unwrap_or_else(|| Arc::new(NoDesaturation)),
        }
    }
}

impl Controller for AttitudeController {
    fn kind(&self) -> i32 {
        CTRL_ATTITUDE
    }

    fn feedback_matrices(&self) -> Option<(Mat3, Mat3)> {
        Some(self.feedback.to_matrices())
    }

    fn validate_sensors(&self, _sensor_types: &[&str]) -> Result<(), String> {
        Ok(())
    }

    fn to_params(&self) -> Vec<f64> {
        let desat_p = self.desaturation.to_params();
        vec![
            f64::from(self.feedforward.kind()),
            f64::from(self.feedback.kind()),
            f64::from(self.allocator.kind()),
            f64::from(self.desaturation.kind()),
            desat_p[0],
            desat_p[1],
            desat_p[2],
        ]
    }
}
