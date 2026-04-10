"""
SALTRO-based trajectory planning and tracking controller.

This module provides a controller wrapper around the ``saltro_py`` backend,
including:

- trajectory planning over a J2000 time horizon,
- conversion of ADCS goal definitions into SALTRO target arrays,
- actuator-order remapping between Python and C++ conventions,
- closed-loop trajectory tracking through the shared ``Trajectory`` container.

The implementation is designed to match the ADCS ``Controller`` interface while
delegating optimization to the SALTRO C++ backend.
"""

__all__ = ["SALTRO"]

import os
import sys
from typing import Optional

import numpy as np

from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.optional_dependencies import get_saltro_module
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

def _ensure_saltro_path() -> str:
    """Ensure the SALTRO build directory is available on ``sys.path``.

    The Python bindings for SALTRO are generated in ``SALTRO/build``. This
    helper appends that directory to ``sys.path`` if it is not already present
    and returns the resolved path.

    :return: Absolute path to the SALTRO build directory.
    :rtype: str
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    saltro_path = os.path.join(parent_dir, "SALTRO", "build")
    if saltro_path not in sys.path:
        sys.path.append(saltro_path)
    return saltro_path


_ensure_saltro_path()


class SALTRO(Controller):
    r"""
    SALTRO trajectory planning and tracking controller.

    This controller computes an optimized trajectory using the ``saltro_py``
    backend and then tracks it via the shared
    :class:`~ADCS.controller.helpers.trajectory.Trajectory` interface.

    Relationship to ADCS framework
    ------------------------------
    The class conforms to :class:`~ADCS.controller.Controller` and therefore
    integrates with :func:`ADCS.simulate` using the standard controller hooks:

    - :meth:`calculate_trajectory` for planning,
    - :meth:`set_active_trajectory` for plan activation,
    - :meth:`find_u` for control evaluation at runtime.

    Time and horizon model
    ----------------------
    Planning is performed over :math:`[t_0, t_1]` in J2000 centuries, where

    .. math::

       t_1 = t_0 + T \cdot c_{\mathrm{sec}\rightarrow\mathrm{cent}},

    with duration :math:`T` in seconds.

    :param est_sat: Estimated satellite model used for constraints and actuator
        definitions.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param planner_settings: SALTRO planner settings bundle.
    :type planner_settings: :class:`~ADCS.controller.saltro.SALTRO_planner_settings.PlannerSettings`
    """

    def __init__(self, est_sat: EstimatedSatellite, planner_settings: PlannerSettings):
        """Construct the SALTRO controller with no active trajectory.

        :param est_sat: Estimated satellite model used by the planner.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: Planner and constraint settings for SALTRO.
        :type planner_settings: :class:`~ADCS.controller.saltro.SALTRO_planner_settings.PlannerSettings`
        :return: None
        :rtype: None
        """
        self.est_sat = est_sat
        self.planner_settings = planner_settings
        self.active_trajectory: Optional[Trajectory] = None

    def set_active_trajectory(self, traj: Trajectory) -> None:
        """Set the trajectory used by :meth:`find_u` for runtime tracking.

        :param traj: Active trajectory object containing times, states,
            controls, and gains.
        :type traj: :class:`~ADCS.controller.helpers.trajectory.Trajectory`
        :return: None
        :rtype: None
        """
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
        r"""Compute an optimized trajectory using the SALTRO backend.

        The method performs the following pipeline:

        1. Build planner time grid in J2000 centuries.
        2. Sample orbit and active goals at each grid point.
        3. Construct SALTRO target arrays:
            - quaternion targets :math:`q_{\mathrm{goal}} \in \mathbb{R}^{4\times N}`
            - body boresight vectors :math:`b_{\mathrm{body}} \in \mathbb{R}^{3\times N}`
        4. Build C++ satellite model and call ``saltro_py.trajOpt``.
        5. Reorder controls/gains from C++ actuator order (MTQ then RW) to
            Python actuator order.
        6. Return a :class:`~ADCS.controller.helpers.trajectory.Trajectory`.

        Target encoding
        ---------------
        Goals are expected in ADCS ``to_ref`` format:

        - Quaternion target: finite ``[q0, q1, q2, q3]``.
        - Vector target: ``[nan, x, y, z]``.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon duration in seconds.
        :type duration: float
        :param x_0: Initial state vector.
        :type x_0: numpy.ndarray
        :param os_0: Initial orbital state.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal timeline used for target generation.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: Enable planner progress prints.
        :type verbose: bool
        :return: Planned trajectory with states, controls, and gains.
        :rtype: :class:`~ADCS.controller.helpers.trajectory.Trajectory`
        :raises ValueError: If pass configuration or target format is invalid.
        :raises RuntimeError: If SALTRO optimization fails.
        :raises ImportError: If ``saltro_py`` cannot be imported.
        """
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
            saltro_py = get_saltro_module()
        except ImportError as exc:
            raise ImportError(str(exc)) from exc

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
        K_flat = np.asarray(K_flat, dtype=np.float64)

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
        r"""Compute control input from the active trajectory at current time.

        The control command is generated by evaluating trajectory tracking at
        :math:`t = \mathrm{os\_hat.J2000}`:

        .. math::

           u(t) = u_\mathrm{traj}(t, x_{\hat{}}).

        :param x_hat: Estimated state vector.
        :type x_hat: numpy.ndarray
        :param sens: Sensor vector (accepted for interface compatibility).
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite (accepted for interface compatibility).
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Orbital state carrying the current J2000 time.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal: Active goal (accepted for interface compatibility).
        :type goal: typing.Optional[:class:`~ADCS.CONOPS.goals.Goal`]
        :return: Control vector in Python actuator ordering.
        :rtype: numpy.ndarray
        :raises RuntimeError: If no active trajectory is set or time is outside
            trajectory validity interval.
        """
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