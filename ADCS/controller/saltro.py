__all__ = ["SALTRO"]

import os
import sys
from typing import Optional

import numpy as np

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

def _ensure_saltro_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    saltro_path = os.path.join(parent_dir, "SALTRO", "build")
    if saltro_path not in sys.path:
        sys.path.append(saltro_path)
    return saltro_path


_ensure_saltro_path()


class SALTRO(Controller):
    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings):
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        self.active_trajectory: Optional[Trajectory] = None

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

        if not getattr(self.planner_settings, "passes", None):
            raise ValueError("SALTRO planner_settings.passes cannot be empty")

        dt = float(self.planner_settings.passes[0].dt)
        if dt <= 0.0:
            raise ValueError(f"SALTRO pass dt must be > 0, got {dt}")
        t_end = float(t_start + duration * TimeConstants.sec2cent)
        n_steps = max(1, int(np.ceil(duration / dt)))
        jtime = np.ascontiguousarray(
            t_start + (dt * TimeConstants.sec2cent) * np.arange(n_steps + 1, dtype=np.float64),
            dtype=np.float64,
        )
        jtime[-1] = t_end

        sim_orbit = Orbit(os_0, t_end, dt=dt, use_J2=True, fast=True, verbose=False)
        q_goal = np.empty((4, jtime.size), dtype=np.float64)
        boresight = np.empty((3, jtime.size), dtype=np.float64)

        for i, t_k in enumerate(jtime):
            os_at_t = sim_orbit.get_os(float(t_k))
            active_goal = goals.get_active_goal(float(t_k), time_units="centuries")
            target_ref, _w_ref = active_goal.to_ref(os_at_t)
            target_ref = np.asarray(target_ref, dtype=np.float64).reshape(4)

            if np.isnan(target_ref[0]):
                if not np.isfinite(target_ref[1:4]).all():
                    raise ValueError("SALTRO vector-goal target must be [nan, x, y, z] with finite x/y/z")
                q_goal[:, i] = target_ref
            else:
                q_ref = normalize(target_ref)
                if not np.isfinite(q_ref).all() or q_ref.shape != (4,):
                    raise ValueError("SALTRO goal must be a finite quaternion or vector-goal [nan, x, y, z]")
                q_goal[:, i] = q_ref

            boresight_name = getattr(active_goal, "boresight_name", None)
            try:
                body_boresight = np.asarray(
                    self.est_sat.get_boresight(boresight_name), dtype=np.float64
                ).reshape(3)
            except (KeyError, ValueError, TypeError, AttributeError):
                body_boresight = np.asarray(self.est_sat.get_boresight(), dtype=np.float64).reshape(3)
            boresight[:, i] = body_boresight

        q_goal = np.ascontiguousarray(q_goal, dtype=np.float64)
        boresight = np.ascontiguousarray(boresight, dtype=np.float64)

        try:
            _ensure_saltro_path()
            import saltro_py
        except ImportError as exc:
            raise ImportError(f"saltro_py not available: {exc}") from exc

        cpp_settings = self.planner_settings.to_cpp()

        cpp_sat = saltro_py.Satellite()
        cpp_sat.setInertia(np.asarray(self.est_sat.J_COM, dtype=np.float64))

        for act in self.est_sat.actuators:
            if isinstance(act, MTQ):
                cpp_sat.addMTQ(np.asarray(act.axis, dtype=np.float64), float(act.u_max))
        for act in self.est_sat.actuators:
            if isinstance(act, RW):
                cpp_sat.addRW(
                    np.asarray(act.axis, dtype=np.float64),
                    float(act.u_max),
                    float(act.J),
                    float(act.h),
                    float(act.h_max),
                )

        r0 = np.asarray(os_0.R, dtype=np.float64).reshape(3) * 1.0e3
        v0 = np.asarray(os_0.V, dtype=np.float64).reshape(3) * 1.0e3
        x0_clean = np.asarray(x_0, dtype=np.float64).reshape(-1)

        ok, Xset, Uset_cpp, K_flat = saltro_py.trajOpt(
            cpp_settings,
            cpp_sat,
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
        K_flat = 0.1 * np.asarray(K_flat, dtype=np.float64)

        n_out = int(Xset.shape[1])
        times = np.linspace(t_start, t_end, n_out, dtype=np.float64)

        cpp_to_py = [i for i, act in enumerate(self.est_sat.actuators) if isinstance(act, MTQ)] + [
            i for i, act in enumerate(self.est_sat.actuators) if isinstance(act, RW)
        ]
        cpp_to_py = np.asarray(cpp_to_py, dtype=int)

        if Uset_cpp.shape[0] == cpp_to_py.size:
            Uset = Uset_cpp[cpp_to_py, :]
        elif Uset_cpp.shape[1] == cpp_to_py.size:
            Uset = Uset_cpp[:, cpp_to_py]
        else:
            raise ValueError(f"Unexpected SALTRO control shape {Uset_cpp.shape}")

        n_red = int(cpp_sat.reducedStateDim)
        if K_flat.shape[1] != n_red * n_out:
            raise ValueError(
                f"Unexpected SALTRO gain shape {K_flat.shape}, expected second dim {n_red * n_out}"
            )

        K_cpp_time = np.zeros((n_out, K_flat.shape[0], n_red), dtype=np.float64)
        for k in range(n_out):
            c0 = k * n_red
            c1 = c0 + n_red
            K_cpp_time[k, :, :] = K_flat[:, c0:c1]
        Kset = -K_cpp_time[:, cpp_to_py, :]

        traj = Trajectory(times, Xset, Uset, Kset, np.zeros(n_out, dtype=np.float64))
        self.active_trajectory = traj
        return traj

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Optional[Goal] = None,
        **kwargs,
    ) -> np.ndarray:
        _ = sens
        _ = est_sat
        _ = goal
        _ = kwargs

        current_time = float(os_hat.J2000)

        if self.active_trajectory is None:
            raise RuntimeError(f"SALTRO: No active trajectory set at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                "SALTRO: Active trajectory expired or not started. "
                f"Current: {current_time}, Traj: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        return self.active_trajectory.compute_tracking_control(current_time, x_hat)