__all__ = ["SALTRO"]

import sys
import os
import numpy as np
from typing import Any, Optional, Tuple

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_helpers import normalize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)


def _get_saltro_py() -> Any:
    import saltro_py
    return saltro_py


def _get_cpp_to_python_control_permutation(actuators):
    mtq_py_indices = [i for i, act in enumerate(actuators) if isinstance(act, MTQ)]
    rw_py_indices = [i for i, act in enumerate(actuators) if isinstance(act, RW)]

    n_mtq = len(mtq_py_indices)
    n_rw = len(rw_py_indices)
    n_total = n_mtq + n_rw

    cpp_to_py = np.zeros(n_total, dtype=int)
    for cpp_idx, py_idx in enumerate(mtq_py_indices):
        cpp_to_py[cpp_idx] = py_idx
    for i, py_idx in enumerate(rw_py_indices):
        cpp_to_py[n_mtq + i] = py_idx

    py_to_cpp = np.zeros(n_total, dtype=int)
    for cpp_idx, py_idx in enumerate(cpp_to_py):
        py_to_cpp[py_idx] = cpp_idx

    return cpp_to_py, py_to_cpp


def _reorder_controls_cpp_to_python(Uset: np.ndarray, actuators) -> np.ndarray:
    cpp_to_py, _ = _get_cpp_to_python_control_permutation(actuators)
    n_ctrl = len(cpp_to_py)

    if Uset.shape[0] == n_ctrl:
        return Uset[cpp_to_py, :]
    if Uset.shape[1] == n_ctrl:
        return Uset[:, cpp_to_py]

    raise ValueError(f"Uset shape {Uset.shape} does not match n_controls={n_ctrl}")


def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _quat_set_basis(body_vec: np.ndarray, eci_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return basis (x, y) spanning quaternion set that aligns body_vec to eci_vec."""
    v = _safe_normalize(body_vec)
    u = _safe_normalize(eci_vec)
    x = np.concatenate(([1.0 + np.dot(v, u)], np.cross(v, u)))
    y = np.concatenate(([0.0], u + v))
    x = _safe_normalize(x)
    y = _safe_normalize(y)
    return x, y


def _closest_quat_in_alignment_set(q_ref: np.ndarray, body_vec: np.ndarray, eci_vec: np.ndarray) -> np.ndarray:
    """Project q_ref onto the vector-goal alignment quaternion set."""
    q_ref = normalize(np.asarray(q_ref, dtype=np.float64).reshape(4))
    body_vec = _safe_normalize(body_vec)
    eci_vec = _safe_normalize(eci_vec)

    dot_vu = float(np.clip(np.dot(body_vec, eci_vec), -1.0, 1.0))

    # Opposite-vector case is degenerate for set-basis formulas; choose a stable 180 deg axis.
    if dot_vu < -1.0 + 1e-10:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(perp, body_vec)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        perp = _safe_normalize(perp - np.dot(perp, body_vec) * body_vec)
        q = np.concatenate(([0.0], perp))
        q = normalize(q)
        if np.dot(q, q_ref) < 0.0:
            q = -q
        return q

    x, y = _quat_set_basis(body_vec, eci_vec)
    if np.linalg.norm(x) < 1e-12 or np.linalg.norm(y) < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if np.dot(q, q_ref) < 0.0:
            q = -q
        return q

    qdx = float(np.dot(q_ref, x))
    qdy = float(np.dot(q_ref, y))
    q = _safe_normalize(qdx * x + qdy * y)
    if np.dot(q, q_ref) < 0.0:
        q = -q
    return normalize(q)

class SALTRO(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings):
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        self.active_trajectory: Optional[Trajectory] = None

    @staticmethod
    def _adcs_orbit_to_saltro_si(r_km: np.ndarray, v_kmps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ADCS orbit vectors [km, km/s] to SALTRO SI units [m, m/s]."""
        r_m = np.asarray(r_km, dtype=np.float64).reshape(3) * 1.0e3
        v_mps = np.asarray(v_kmps, dtype=np.float64).reshape(3) * 1.0e3
        return r_m, v_mps

    def _build_goal_arrays(
        self,
        goals: GoalList,
        sim_orbit: Orbit,
        jtime: np.ndarray,
        q_seed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build q_goal (4,N) and boresight (3,N) arrays for SALTRO."""
        N = int(jtime.size)
        q_goal = np.zeros((4, N), dtype=np.float64)
        boresight = np.zeros((3, N), dtype=np.float64)

        default_boresight = np.asarray(self.est_sat.get_boresight(), dtype=np.float64).reshape(3)
        q_prev = normalize(np.asarray(q_seed, dtype=np.float64).reshape(4))

        for i, t in enumerate(jtime):
            os_at_t = sim_orbit.get_os(float(t))
            target, _w_ref = goals.to_ref(float(t), os_at_t, time_units="centuries")
            target = np.asarray(target, dtype=np.float64).reshape(-1)

            active_goal: Goal = goals.get_active_goal(float(t), time_units="centuries")
            boresight_name = getattr(active_goal, "boresight_name", None)

            if boresight_name is not None and boresight_name in self.est_sat.boresight:
                body_vec = np.asarray(self.est_sat.get_boresight(boresight_name), dtype=np.float64).reshape(3)
            else:
                body_vec = default_boresight

            # SALTRO natively accepts mixed goal columns:
            # - Quaternion: [q0, qx, qy, qz]
            # - Vector goal: [nan, x, y, z]
            if target.size < 4:
                raise ValueError(f"Goal vector must have at least 4 elements, got shape {target.shape}")

            q_ref = np.asarray(target[:4], dtype=np.float64)
            if np.isnan(q_ref[0]):
                # Vector-goal format [nan, x, y, z]: convert to closest quaternion
                # in the alignment set to preserve roll-free semantics and continuity.
                tail_norm = np.linalg.norm(q_ref[1:4])
                if tail_norm > 1e-12:
                    eci_vec = q_ref[1:4] / tail_norm
                    q_ref = _closest_quat_in_alignment_set(q_prev, body_vec, eci_vec)
                else:
                    q_ref = q_prev.copy()
            else:
                # Quaternion-goal format [q0, qx, qy, qz]: normalize quaternion.
                q_ref = normalize(q_ref)
                if np.dot(q_ref, q_prev) < 0.0:
                    q_ref = -q_ref

            q_goal[:, i] = q_ref
            boresight[:, i] = body_vec
            q_prev = q_ref

        return q_goal, boresight

    def settings_to_cpp(self, settings: PlannerSettings):
        cpp_settings = settings.to_cpp()
        return cpp_settings
    
    def satellite_to_cpp(self, satellite: EstimatedSatellite):
        saltro_py = _get_saltro_py()
        cpp_sat = saltro_py.Satellite()
        # SALTRO expects inertia about COM.
        cpp_sat.setInertia(np.asarray(satellite.J_COM, dtype=np.float64))

        # Keep C++ ordering deterministic: MTQ first, then RW.
        for act in satellite.actuators:
            if isinstance(act, MTQ):
                cpp_sat.addMTQ(np.asarray(act.axis, dtype=np.float64), float(act.u_max))

        for act in satellite.actuators:
            if isinstance(act, RW):
                cpp_sat.addRW(
                    np.asarray(act.axis, dtype=np.float64),
                    float(act.u_max),
                    float(act.J),
                    float(act.h),
                    float(act.h_max),
                )

        return cpp_sat

    @staticmethod
    def _reshape_saltro_gains(K_flat: np.ndarray, n_red: int, N: int) -> np.ndarray:
        """Convert SALTRO K from (nu, n_red*N) to (N, nu, n_red)."""
        K_flat = np.asarray(K_flat, dtype=np.float64)
        n_u = int(K_flat.shape[0])
        K_time = np.zeros((N, n_u, n_red), dtype=np.float64)

        for k in range(N):
            col0 = k * n_red
            col1 = col0 + n_red
            K_time[k, :, :] = K_flat[:, col0:col1]

        return K_time

    def set_active_trajectory(self, traj: Trajectory) -> None:
        self.active_trajectory = traj

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
    ) -> Trajectory:
        if verbose:
            print(f"SALTRO planning: start={t_start:.8f} centuries, duration={duration:.3f} s")

        if len(self.planner_settings.passes) == 0:
            raise ValueError("SALTRO planner_settings.passes cannot be empty")

        dt = float(self.planner_settings.passes[0].dt)
        if dt <= 0.0:
            raise ValueError(f"SALTRO pass dt must be > 0, got {dt}")

        N = int(np.ceil(duration / dt)) + 1
        jtime = np.ascontiguousarray(
            t_start + np.arange(N, dtype=np.float64) * dt * TimeConstants.sec2cent,
            dtype=np.float64,
        )

        t_end = float(jtime[-1])
        t_end_buffered = t_end + 10.0 * dt * TimeConstants.sec2cent
        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt, use_J2=True, fast=False)

        x0_clean = np.ascontiguousarray(np.asarray(x_0, dtype=np.float64).reshape(-1), dtype=np.float64)

        q_goal, boresight = self._build_goal_arrays(
            goals=goals,
            sim_orbit=sim_orbit,
            jtime=jtime,
            q_seed=x0_clean[3:7],
        )

        saltro_py = _get_saltro_py()
        cpp_settings = self.settings_to_cpp(self.planner_settings)
        cpp_satellite = self.satellite_to_cpp(self.est_sat)

        r0, v0 = self._adcs_orbit_to_saltro_si(os_0.R, os_0.V)

        ok, Xset, Uset_cpp, K_flat = saltro_py.trajOpt(
            cpp_settings,
            cpp_satellite,
            x0_clean,
            r0,
            v0,
            jtime,
            q_goal,
            boresight,
        )
        if not ok:
            raise RuntimeError("SALTRO trajOpt returned ok=False")

        Xset = np.asarray(Xset, dtype=np.float64)
        Uset_cpp = np.asarray(Uset_cpp, dtype=np.float64)
        K_flat = np.asarray(K_flat, dtype=np.float64)

        N_out = int(Xset.shape[1])
        lqr_times = np.ascontiguousarray(jtime[:N_out], dtype=np.float64)

        Uset = _reorder_controls_cpp_to_python(Uset_cpp, self.est_sat.actuators)

        n_red = int(cpp_satellite.reducedStateDim)
        expected_cols = n_red * N_out
        if K_flat.shape[1] != expected_cols:
            raise ValueError(
                f"Unexpected SALTRO gain shape {K_flat.shape}, expected second dim {expected_cols} "
                f"(n_red={n_red}, N_out={N_out})"
            )

        K_cpp_time = self._reshape_saltro_gains(K_flat, n_red=n_red, N=N_out)
        cpp_to_py, _ = _get_cpp_to_python_control_permutation(self.est_sat.actuators)
        # SALTRO backward pass uses optimizer form u = u_nom + K*dx + d,
        # while Trajectory tracking uses u = u_ref - K*dx. Negate once here
        # so closed-loop feedback has the correct sign in Python.
        Kset = -K_cpp_time[:, cpp_to_py, :]

        # SALTRO pybind does not expose S/cost-to-go yet.
        Sset = np.zeros(N_out, dtype=np.float64)
        return Trajectory(lqr_times, Xset, Uset, Kset, Sset)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Optional[Goal] = None,
    ) -> np.ndarray:
        _ = sens
        _ = est_sat
        _ = goal

        current_time = float(os_hat.J2000)

        if self.active_trajectory is None:
            raise RuntimeError(f"SALTRO: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                "SALTRO: Active trajectory expired or not started. "
                f"Current: {current_time}, "
                f"Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        return self.active_trajectory.compute_tracking_control(current_time, x_hat)