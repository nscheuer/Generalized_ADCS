use std::sync::Arc;

use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use crate::controllers::{
    init_controller_work, prepare_controllers, step_controller, BCross, Controller,
    ControllerTable, SensorConfig,
};

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn test_bcross_validation_and_requires_mag_and_gyro() {
    assert!(BCross::new(-1.0).is_err());
    let bcross = BCross::new(0.05).unwrap();
    assert!(bcross.validate_sensors(&["Magnetometer"]).is_err());
    assert!(bcross.validate_sensors(&["Magnetometer", "Gyroscope"]).is_ok());
}

#[test]
fn test_bcross_damps_angular_rate_in_loop() {
    let bcross: Arc<dyn Controller> = Arc::new(BCross::new(0.05).unwrap());
    let mut table = ControllerTable::new();
    let idx = table.register(Some(bcross));
    let rt = prepare_controllers(
        &[],
        &["Magnetometer", "Gyroscope"],
        vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        vec![1.0, 1.0, 1.0],
        Some(&mut table),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let mut sc = SensorConfig::default();
    sc.code.push(1);
    sc.orientation.push([1.0, 0.0, 0.0, 0.0]);
    sc.enabled.push(true);
    sc.offsets.push(0);
    sc.code.push(0);
    sc.orientation.push([1.0, 0.0, 0.0, 0.0]);
    sc.enabled.push(true);
    sc.offsets.push(3);

    let mut ctrl_work = init_controller_work();
    let mut wheel_cmd = vec![];
    let mut dipole_cmd = vec![0.0; 3];

    let mut q = [1.0, 0.0, 0.0, 0.0];
    let mut omega = [0.05, -0.04, 0.03];
    let initial_norm = norm3(omega);
    let inertia = Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.1));
    let inv_inertia = inertia.try_inverse().unwrap();
    let b_inertial = Vector3::new(2.0e-5, -1.5e-5, 3.0e-5);
    let dt = 0.25;

    for _ in 0..480 {
        let uq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        let b_body_v = uq.inverse_transform_vector(&b_inertial);
        let readings = [
            b_body_v[0],
            b_body_v[1],
            b_body_v[2],
            omega[0],
            omega[1],
            omega[2],
        ];
        step_controller(
            idx,
            dt,
            q,
            omega,
            [0.0; 4],
            [0.0; 3],
            &[],
            &readings,
            &sc,
            &rt,
            &mut ctrl_work,
            &mut wheel_cmd,
            &mut dipole_cmd,
        );
        let m = Vector3::new(dipole_cmd[0], dipole_cmd[1], dipole_cmd[2]);
        let w_v = Vector3::new(omega[0], omega[1], omega[2]);
        let torque = m.cross(&b_body_v);
        let wdot = inv_inertia * (torque - w_v.cross(&(inertia * w_v)));
        omega = [
            omega[0] + dt * wdot[0],
            omega[1] + dt * wdot[1],
            omega[2] + dt * wdot[2],
        ];
        let dq = quat_mul(q, [0.0, omega[0], omega[1], omega[2]]);
        for i in 0..4 {
            q[i] += 0.5 * dt * dq[i];
        }
        let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        for i in 0..4 {
            q[i] /= qn;
        }
    }

    assert!(norm3(omega) < initial_norm);
}
