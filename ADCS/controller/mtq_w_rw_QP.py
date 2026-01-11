__all__ = ["MTQ_w_RW_QP"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller, MTQ_w_RW_LP
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_QP(MTQ_w_RW_LP):
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray | list = np.zeros(3)) -> None:
        super().__init__(est_sat=est_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=h_target)

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> tuple[np.ndarray, np.ndarray, float]:
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1) Setup matrices exactly like before
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        # RW torque map: tau_rw = A_rw * u_rw
        if rws:
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])
            u_rw_lims = np.array([rw.u_max for rw in rws], dtype=float)
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0, dtype=float)

        # MTQ torque map: tau_mtq = ( -skew(B) * A_axes ) * u_mtq
        if mtqs:
            b_skew = -skewsym(b_body)
            A_mtq_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])
            A_mtq = b_skew @ A_mtq_axes
            u_mtq_lims = np.array([m.u_max for m in mtqs], dtype=float)
        else:
            A_mtq = np.zeros((3, 0))
            u_mtq_lims = np.zeros(0, dtype=float)

        A_total = np.hstack([A_rw, A_mtq])  # (3, n_act)
        n_act = A_total.shape[1]

        n_rw = len(rws)
        n_mtq = len(mtqs)

        if n_act == 0:
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        # 2) Bounds: [-u_max, +u_max]
        lb = np.concatenate([-u_rw_lims, -u_mtq_lims]) if n_act else np.zeros(0)
        ub = np.concatenate([ u_rw_lims,  u_mtq_lims]) if n_act else np.zeros(0)

        # 3) Solve bounded least squares: minimize ||A u - tau_des||^2
        # Note: lsq_linear can handle rank-deficient A robustly.
        res = lsq_linear(A_total, tau_des, bounds=(lb, ub), method="trf")

        if not res.success:
            # fall back: best effort (still keep shape)
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        u_sol = res.x  # length n_act

        # 4) Compute alpha as "how much of tau_des magnitude you achieved along its direction"
        tau_ach = A_total @ u_sol
        tau_hat = tau_des / (t_mag + 1e-12)
        T_along = float(np.dot(tau_ach, tau_hat))          # signed magnitude along desired direction
        alpha = max(0.0, T_along / (t_mag + 1e-12))        # normalize by requested magnitude, clamp to [0, inf)

        # Split commands
        u_rw_cmd = u_sol[:n_rw]
        u_mtq_cmd = u_sol[n_rw:n_rw + n_mtq]

        return u_rw_cmd, u_mtq_cmd, alpha