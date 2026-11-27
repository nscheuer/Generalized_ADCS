use std::sync::Arc;

use nalgebra::{DMatrix, SymmetricEigen};

use super::attitude_controller::{step as attitude_step, AttitudeController};
use super::bcross::step as bcross_step;
use super::bdot::step as bdot_step;
use super::controller::{
    Controller, NoController, SensorConfig, CTRL_ATTITUDE, CTRL_BCROSS, CTRL_BDOT, CTRL_NONE,
};
use super::feedback::Mat3;

pub const CTRL_PARAM_WIDTH: usize = 7;

fn compute_allocation_matrices(
    a_mat: &DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let n_w = a_mat.ncols();
    if n_w == 0 {
        return (
            DMatrix::zeros(0, 3),
            DMatrix::zeros(0, 0),
            DMatrix::zeros(0, 0),
        );
    }
    let a_pinv = a_mat
        .clone()
        .pseudo_inverse(1e-15)
        .unwrap_or_else(|_| DMatrix::zeros(n_w, 3));
    let null_proj = DMatrix::identity(n_w, n_w) - &a_pinv * a_mat;

    // Orthonormal basis for null(A) via eigendecomposition of A^T A (symmetric n_w x n_w).
    let ata = a_mat.transpose() * a_mat;
    let eig = SymmetricEigen::new(ata);
    let max_lam = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let lam_tol = 1e-12 * max_lam.max(1.0);
    let mut null_cols = Vec::new();
    for (idx, &lam) in eig.eigenvalues.iter().enumerate() {
        if lam <= lam_tol {
            null_cols.push(idx);
        }
    }
    let mut null_basis = DMatrix::zeros(n_w, null_cols.len());
    for (col_out, &col_eig) in null_cols.iter().enumerate() {
        for r in 0..n_w {
            null_basis[(r, col_out)] = eig.eigenvectors[(r, col_eig)];
        }
    }

    (a_pinv, null_proj, null_basis)
}

#[derive(Clone)]
pub struct ControllerTable {
    pub kinds: Vec<i32>,
    pub params: Vec<[f64; CTRL_PARAM_WIDTH]>,
    pub kp: Vec<Mat3>,
    pub kd: Vec<Mat3>,
    pub objects: Vec<Arc<dyn Controller>>,
    pub attitude_controllers: Vec<Option<AttitudeController>>,
}

impl Default for ControllerTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ControllerTable {
    pub fn new() -> Self {
        Self {
            kinds: vec![CTRL_NONE],
            params: vec![[0.0; CTRL_PARAM_WIDTH]],
            kp: vec![[[0.0; 3]; 3]],
            kd: vec![[[0.0; 3]; 3]],
            objects: vec![Arc::new(NoController)],
            attitude_controllers: vec![None],
        }
    }

    pub fn register(&mut self, controller: Option<Arc<dyn Controller>>) -> usize {
        self.register_with_attitude(controller, None)
    }

    pub fn register_attitude(&mut self, controller: AttitudeController) -> usize {
        let arc: Arc<dyn Controller> = Arc::new(controller.clone());
        self.register_with_attitude(Some(arc), Some(controller))
    }

    fn register_with_attitude(
        &mut self,
        controller: Option<Arc<dyn Controller>>,
        attitude: Option<AttitudeController>,
    ) -> usize {
        let Some(ctrl) = controller else {
            return 0;
        };
        if ctrl.kind() == CTRL_NONE {
            return 0;
        }
        let mut p = [0.0; CTRL_PARAM_WIDTH];
        let raw = ctrl.to_params();
        let copy_len = raw.len().min(CTRL_PARAM_WIDTH);
        p[..copy_len].copy_from_slice(&raw[..copy_len]);

        let (kp_mat, kd_mat) = if let Some(ref att) = attitude {
            att.feedback.to_matrices()
        } else {
            ctrl.feedback_matrices()
                .unwrap_or(([[0.0; 3]; 3], [[0.0; 3]; 3]))
        };

        let idx = self.kinds.len();
        self.kinds.push(ctrl.kind());
        self.params.push(p);
        self.kp.push(kp_mat);
        self.kd.push(kd_mat);
        self.objects.push(ctrl);
        self.attitude_controllers.push(attitude);
        idx
    }
}

#[derive(Clone, Debug)]
pub struct ControllerRuntime {
    pub enabled: bool,
    pub kinds: Vec<i32>,
    pub params: Vec<[f64; CTRL_PARAM_WIDTH]>,
    pub kp: Vec<Mat3>,
    pub kd: Vec<Mat3>,
    pub mtq_axes: Vec<[f64; 3]>,
    pub mtq_max_dipole: Vec<f64>,
    pub wheel_axes: Vec<[f64; 3]>,
    pub wheel_max_torque: Vec<f64>,
    pub a_pinv: DMatrix<f64>,
    pub null_basis: DMatrix<f64>,
    pub null_proj: DMatrix<f64>,
    pub total_inertia: Mat3,
    pub bus_inertia: Mat3,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_controllers(
    controllers: &[Option<Arc<dyn Controller>>],
    sensor_types: &[&str],
    mtq_axes: Vec<[f64; 3]>,
    mtq_max_dipole: Vec<f64>,
    table: Option<&mut ControllerTable>,
    wheel_axes: Option<Vec<[f64; 3]>>,
    wheel_max_torque: Option<Vec<f64>>,
    total_inertia: Option<Mat3>,
    bus_inertia: Option<Mat3>,
) -> Result<ControllerRuntime, String> {
    let mut local_table = ControllerTable::new();
    let tbl = table.unwrap_or(&mut local_table);

    for ctrl in controllers {
        tbl.register(ctrl.clone());
    }
    for ctrl in &tbl.objects {
        ctrl.validate_sensors(sensor_types)?;
    }
    let enabled = tbl.kinds.iter().any(|&k| k != CTRL_NONE);

    let w_axes = wheel_axes.unwrap_or_default();
    let w_max_t = wheel_max_torque.unwrap_or_default();
    let n_w = w_axes.len();
    let a_mat = if n_w > 0 {
        DMatrix::from_fn(3, n_w, |r, c| w_axes[c][r])
    } else {
        DMatrix::zeros(3, 0)
    };
    let (a_pinv, null_proj, null_basis) = compute_allocation_matrices(&a_mat);

    let t_iner = total_inertia.unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let b_iner = bus_inertia.unwrap_or(t_iner);

    Ok(ControllerRuntime {
        enabled,
        kinds: tbl.kinds.clone(),
        params: tbl.params.clone(),
        kp: tbl.kp.clone(),
        kd: tbl.kd.clone(),
        mtq_axes,
        mtq_max_dipole,
        wheel_axes: w_axes,
        wheel_max_torque: w_max_t,
        a_pinv,
        null_basis,
        null_proj,
        total_inertia: t_iner,
        bus_inertia: b_iner,
    })
}

pub fn init_controller_work() -> [f64; 4] {
    [0.0; 4]
}

pub fn sync_actuator_enable(
    rw_enabled: &[bool],
    mtq_enabled: &[bool],
    base_wheel_max: &[f64],
    base_mtq_max: &[f64],
    ctrl_rt: &mut ControllerRuntime,
    wheel_commands: &mut [f64],
    dipole_commands: &mut [f64],
) {
    for (i, &en) in mtq_enabled.iter().enumerate() {
        if en {
            ctrl_rt.mtq_max_dipole[i] = base_mtq_max[i];
        } else {
            ctrl_rt.mtq_max_dipole[i] = 0.0;
            dipole_commands[i] = 0.0;
        }
    }

    let n_w = rw_enabled.len();
    if n_w == 0 {
        return;
    }
    let mut active_wheels = 0usize;
    for (i, &en) in rw_enabled.iter().enumerate() {
        if en {
            ctrl_rt.wheel_max_torque[i] = base_wheel_max[i];
            active_wheels += 1;
        } else {
            ctrl_rt.wheel_max_torque[i] = 0.0;
            wheel_commands[i] = 0.0;
        }
    }

    if active_wheels == 0 {
        ctrl_rt.a_pinv.fill(0.0);
        ctrl_rt.null_proj.fill(0.0);
        ctrl_rt.null_basis.fill(0.0);
        return;
    }

    let a_mat = DMatrix::from_fn(3, n_w, |r, c| {
        if rw_enabled[c] {
            ctrl_rt.wheel_axes[c][r]
        } else {
            0.0
        }
    });
    let (a_pinv, null_proj, full_null_basis) = compute_allocation_matrices(&a_mat);
    ctrl_rt.a_pinv = a_pinv;
    ctrl_rt.null_proj = null_proj;
    if ctrl_rt.null_basis.ncols() > 0 {
        ctrl_rt.null_basis.fill(0.0);
        let cols = ctrl_rt.null_basis.ncols();
        let avail = full_null_basis.ncols();
        let take = cols.min(avail);
        for c in 0..take {
            for (r, &en) in rw_enabled.iter().enumerate() {
                if en {
                    ctrl_rt.null_basis[(r, c)] = full_null_basis[(r, c)];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn step_controller(
    ctrl_idx: usize,
    dt_ctrl: f64,
    q: [f64; 4],
    omega: [f64; 3],
    q_goal: [f64; 4],
    w_goal: [f64; 3],
    wheel_momenta: &[f64],
    readings: &[f64],
    sensor_config: &SensorConfig,
    ctrl_rt: &ControllerRuntime,
    ctrl_work: &mut [f64],
    wheel_commands: &mut [f64],
    dipole_commands: &mut [f64],
) -> ([f64; 3], [f64; 4], [f64; 3], [f64; 3]) {
    let nan_q = [f64::NAN; 4];
    let nan_3 = [f64::NAN; 3];
    let kind = ctrl_rt.kinds.get(ctrl_idx).copied().unwrap_or(-1);

    if kind == CTRL_NONE {
        dipole_commands.fill(0.0);
        wheel_commands.fill(0.0);
        return (nan_3, nan_q, nan_3, nan_3);
    }

    if kind == CTRL_BDOT {
        wheel_commands.fill(0.0);
        if !dipole_commands.is_empty() {
            bdot_step(
                ctrl_rt.params[ctrl_idx][0],
                dt_ctrl,
                readings,
                sensor_config,
                &ctrl_rt.mtq_axes,
                &ctrl_rt.mtq_max_dipole,
                ctrl_work,
                dipole_commands,
            );
        }
        return (nan_3, nan_q, nan_3, nan_3);
    }

    if kind == CTRL_BCROSS {
        wheel_commands.fill(0.0);
        if !dipole_commands.is_empty() {
            bcross_step(
                ctrl_rt.params[ctrl_idx][0],
                readings,
                sensor_config,
                &ctrl_rt.mtq_axes,
                &ctrl_rt.mtq_max_dipole,
                dipole_commands,
            );
        }
        return (nan_3, nan_q, nan_3, nan_3);
    }

    if kind == CTRL_ATTITUDE {
        let p = &ctrl_rt.params[ctrl_idx];
        return attitude_step(
            q,
            omega,
            q_goal,
            w_goal,
            wheel_momenta,
            &ctrl_rt.total_inertia,
            &ctrl_rt.bus_inertia,
            &ctrl_rt.wheel_axes,
            &ctrl_rt.wheel_max_torque,
            &ctrl_rt.a_pinv,
            &ctrl_rt.null_basis,
            &ctrl_rt.null_proj,
            p[0] as i32,
            p[1] as i32,
            p[2] as i32,
            p[3] as i32,
            &ctrl_rt.kp[ctrl_idx],
            &ctrl_rt.kd[ctrl_idx],
            p[4],
            p[5],
            p[6],
            readings,
            sensor_config,
            &ctrl_rt.mtq_axes,
            &ctrl_rt.mtq_max_dipole,
            wheel_commands,
            dipole_commands,
        );
    }

    (nan_3, nan_q, nan_3, nan_3)
}
