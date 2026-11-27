use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};

use crate::controllers::{
    init_controller_work, prepare_controllers, step_controller, AttitudeController,
    ControllerTable, HoganSchaubManagement, L2Allocator, LInfinityAllocator, MRPFeedbackLinearized,
    NoDesaturation, Passthrough, SensorConfig,
};

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn rk4_step(
    q: &mut [f64; 4],
    omega: &mut [f64; 3],
    wheel_h: &mut [f64],
    wheel_cmd: &[f64],
    wheel_axes: &[[f64; 3]],
    inertia: &Matrix3<f64>,
    inv_bus: &Matrix3<f64>,
    dt: f64,
) {
    let mut tau_rw = Vector3::zeros();
    for (ax, &cmd) in wheel_axes.iter().zip(wheel_cmd.iter()) {
        tau_rw += Vector3::new(ax[0], ax[1], ax[2]) * cmd;
    }

    let rhs = |qs: [f64; 4], ws: [f64; 3], hs: &[f64]| -> ([f64; 4], Vector3<f64>, Vec<f64>) {
        let dq = quat_mul([0.0, -ws[0], -ws[1], -ws[2]], qs);
        let qdot = [0.5 * dq[0], 0.5 * dq[1], 0.5 * dq[2], 0.5 * dq[3]];
        let wv = Vector3::new(ws[0], ws[1], ws[2]);
        let mut h_tot = inertia * wv;
        for (ax, &h) in wheel_axes.iter().zip(hs.iter()) {
            h_tot += Vector3::new(ax[0], ax[1], ax[2]) * h;
        }
        let wdot = inv_bus * (tau_rw - wv.cross(&h_tot));
        let hdot: Vec<f64> = wheel_cmd.iter().map(|&c| -c).collect();
        (qdot, wdot, hdot)
    };

    let (k1q, k1w, k1h) = rhs(*q, *omega, wheel_h);
    let q2 = [
        q[0] + 0.5 * dt * k1q[0],
        q[1] + 0.5 * dt * k1q[1],
        q[2] + 0.5 * dt * k1q[2],
        q[3] + 0.5 * dt * k1q[3],
    ];
    let w2 = [
        omega[0] + 0.5 * dt * k1w[0],
        omega[1] + 0.5 * dt * k1w[1],
        omega[2] + 0.5 * dt * k1w[2],
    ];
    let h2: Vec<f64> = wheel_h
        .iter()
        .zip(&k1h)
        .map(|(h, k)| h + 0.5 * dt * k)
        .collect();
    let (k2q, k2w, k2h) = rhs(q2, w2, &h2);

    let q3 = [
        q[0] + 0.5 * dt * k2q[0],
        q[1] + 0.5 * dt * k2q[1],
        q[2] + 0.5 * dt * k2q[2],
        q[3] + 0.5 * dt * k2q[3],
    ];
    let w3 = [
        omega[0] + 0.5 * dt * k2w[0],
        omega[1] + 0.5 * dt * k2w[1],
        omega[2] + 0.5 * dt * k2w[2],
    ];
    let h3: Vec<f64> = wheel_h
        .iter()
        .zip(&k2h)
        .map(|(h, k)| h + 0.5 * dt * k)
        .collect();
    let (k3q, k3w, k3h) = rhs(q3, w3, &h3);

    let q4 = [
        q[0] + dt * k3q[0],
        q[1] + dt * k3q[1],
        q[2] + dt * k3q[2],
        q[3] + dt * k3q[3],
    ];
    let w4 = [
        omega[0] + dt * k3w[0],
        omega[1] + dt * k3w[1],
        omega[2] + dt * k3w[2],
    ];
    let h4: Vec<f64> = wheel_h
        .iter()
        .zip(&k3h)
        .map(|(h, k)| h + dt * k)
        .collect();
    let (k4q, k4w, k4h) = rhs(q4, w4, &h4);

    for i in 0..4 {
        q[i] += (dt / 6.0) * (k1q[i] + 2.0 * k2q[i] + 2.0 * k3q[i] + k4q[i]);
    }
    let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    for i in 0..4 {
        q[i] /= qn;
    }
    for i in 0..3 {
        omega[i] += (dt / 6.0) * (k1w[i] + 2.0 * k2w[i] + 2.0 * k3w[i] + k4w[i]);
    }
    for i in 0..wheel_h.len() {
        wheel_h[i] += (dt / 6.0) * (k1h[i] + 2.0 * k2h[i] + 2.0 * k3h[i] + k4h[i]);
    }
}

#[test]
fn test_attitude_controller_tracks_quaternion_goal() {
    let inertia_arr = [[0.15, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.0, 0.08]];
    let inertia = Matrix3::from_diagonal(&Vector3::new(0.15, 0.12, 0.08));
    let inv_bus = inertia.try_inverse().unwrap();
    let wheel_axes = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let wheel_max = vec![0.2, 0.2, 0.2];

    let ctrl = AttitudeController::new(
        Arc::new(MRPFeedbackLinearized::new(0.5, 1.4).unwrap()),
        Some(Arc::new(Passthrough)),
        Some(Arc::new(L2Allocator)),
        Some(Arc::new(NoDesaturation)),
    );

    let mut table = ControllerTable::new();
    let idx = table.register_attitude(ctrl);
    let rt = prepare_controllers(
        &[],
        &[],
        vec![],
        vec![],
        Some(&mut table),
        Some(wheel_axes.clone()),
        Some(wheel_max),
        Some(inertia_arr),
        Some(inertia_arr),
    )
    .unwrap();

    let mut q = [(PI / 6.0).cos(), (PI / 6.0).sin(), 0.0, 0.0];
    let mut omega = [0.05, -0.04, 0.03];
    let mut wheel_h = vec![0.0; 3];
    let target_q = [1.0, 0.0, 0.0, 0.0];
    let target_w = [0.0, 0.0, 0.0];

    let sc = SensorConfig::default();
    let mut ctrl_work = init_controller_work();
    let mut wheel_cmd = vec![0.0; 3];
    let mut dipole_cmd = vec![];
    let dt = 0.05;

    for _ in 0..400 {
        step_controller(
            idx,
            dt,
            q,
            omega,
            target_q,
            target_w,
            &wheel_h,
            &[],
            &sc,
            &rt,
            &mut ctrl_work,
            &mut wheel_cmd,
            &mut dipole_cmd,
        );
        rk4_step(
            &mut q,
            &mut omega,
            &mut wheel_h,
            &wheel_cmd,
            &wheel_axes,
            &inertia,
            &inv_bus,
            dt,
        );
    }

    let dot = q[0] * target_q[0] + q[1] * target_q[1] + q[2] * target_q[2] + q[3] * target_q[3];
    let w_norm = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    assert!(dot.abs() > 0.999);
    assert!(w_norm < 1e-3);
}

#[test]
fn test_attitude_controller_4_wheels_linfinity_and_hogan_schaub() {
    let s3 = 1.0 / 3.0_f64.sqrt();
    let wheel_axes = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [s3, s3, s3],
    ];
    let wheel_max = vec![0.2; 4];
    let inertia_arr = [[0.15, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.0, 0.08]];
    let inertia = Matrix3::from_diagonal(&Vector3::new(0.15, 0.12, 0.08));
    let inv_bus = inertia.try_inverse().unwrap();

    let ctrl = AttitudeController::new(
        Arc::new(MRPFeedbackLinearized::new(0.5, 1.4).unwrap()),
        Some(Arc::new(Passthrough)),
        Some(Arc::new(LInfinityAllocator)),
        Some(Arc::new(HoganSchaubManagement::new(0.5, 0.0, 0.0).unwrap())),
    );

    let mut table = ControllerTable::new();
    let idx = table.register_attitude(ctrl);
    let rt = prepare_controllers(
        &[],
        &[],
        vec![],
        vec![],
        Some(&mut table),
        Some(wheel_axes.clone()),
        Some(wheel_max),
        Some(inertia_arr),
        Some(inertia_arr),
    )
    .unwrap();

    let mut wheel_h: Vec<f64> = (0..4).map(|i| 0.4 * rt.null_basis[(i, 0)]).collect();
    let h0_norm = wheel_h.iter().map(|h| h * h).sum::<f64>().sqrt();
    let mut q = [1.0, 0.0, 0.0, 0.0];
    let mut omega = [0.0; 3];

    let sc = SensorConfig::default();
    let mut ctrl_work = init_controller_work();
    let mut wheel_cmd = vec![0.0; 4];
    let mut dipole_cmd = vec![];
    let dt = 0.05;

    for _ in 0..300 {
        step_controller(
            idx,
            dt,
            q,
            omega,
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            &wheel_h,
            &[],
            &sc,
            &rt,
            &mut ctrl_work,
            &mut wheel_cmd,
            &mut dipole_cmd,
        );
        rk4_step(
            &mut q,
            &mut omega,
            &mut wheel_h,
            &wheel_cmd,
            &wheel_axes,
            &inertia,
            &inv_bus,
            dt,
        );
    }

    let h_final_norm = wheel_h.iter().map(|h| h * h).sum::<f64>().sqrt();
    assert!(h_final_norm < 0.05 * h0_norm);
}
