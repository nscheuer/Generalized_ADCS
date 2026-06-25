__all__ = ["SatelliteAgent"]

import time
import numpy as np
from typing import Optional
from scipy.integrate import solve_ivp

from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.simresults import RunResults
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


class SatelliteAgent:
    r"""
    Self-contained per-satellite simulation stepper.

    A ``SatelliteAgent`` owns one satellite's mutable simulation state (true
    state ``x``, last control ``u``, estimate ``x_hat``/``os_hat``) and its
    :class:`~ADCS.helpers.simresults.RunResults` accumulator, plus references to
    that satellite's controller, estimator(s) and goal timeline. One call to
    :meth:`step` advances the satellite by ``dt`` exactly as the historical
    single-satellite loop in :func:`ADCS.simulate.simulate` did:
    sense -> orbit-estimate -> attitude-estimate -> goal -> control ->
    integrate attitude dynamics -> record.

    The agent does **not** own the orbit. The caller supplies the environment
    orbital states ``os_k`` (current) and ``os_kp1`` (next) for the step, so the
    same agent works whether the orbit is precomputed (single-satellite
    :func:`~ADCS.simulate.simulate`) or batched across a constellation
    (formation simulation). Goals may close over a shared formation world to
    implement neighbor-relative pointing; the agent simply evaluates
    ``goal.to_ref`` / ``goal.error`` and is agnostic to that.

    :param x: Initial true satellite state vector (length ``satellite.state_len``).
    :param satellite: The true plant model.
    :param est_satellite: Estimated-satellite model used by estimator/controller.
    :param controller: Control law (or ``None``).
    :param estimator: Attitude estimator (or ``None``).
    :param orbit_estimator: Orbit estimator (or ``None``).
    :param goal_list: Time-varying :class:`~ADCS.CONOPS.goallist.GoalList`.
    :param sat_id: Optional identifier (used as the run id in formation runs).
    """

    def __init__(
        self,
        x: np.ndarray,
        satellite,
        est_satellite=None,
        controller=None,
        estimator=None,
        orbit_estimator=None,
        goal_list=None,
        sat_id: Optional[object] = None,
    ) -> None:
        self.satellite = satellite
        self.est_satellite = est_satellite
        self.controller = controller
        self.estimator = estimator
        self.orbit_estimator = orbit_estimator
        self.goal_list = goal_list
        self.sat_id = sat_id

        self.x = np.asarray(x, dtype=float).copy()
        self.u = np.zeros(satellite.control_len)
        self.x_hat = None
        if estimator is not None and est_satellite is not None:
            self.x_hat = np.empty(est_satellite.state_len)
        self.os_hat = None

        # Jacobians are only needed when an estimator is in the loop.
        self._skip_jac = (estimator is None)

        self.results = RunResults(satellite=satellite, est_satellite=est_satellite)

    def step(self, k: int, J2000_k: float, os_k: Orbital_State, os_kp1: Orbital_State) -> np.ndarray:
        r"""
        Advance this satellite by one timestep and record the result.

        :param k: Zero-based step index (recorded position).
        :param J2000_k: Current epoch in Julian centuries since J2000.
        :param os_k: Environment orbital state at ``J2000_k``.
        :param os_kp1: Environment orbital state at the next step (for the
            attitude integrator's orbital interpolation).
        :return: The updated true state vector ``x``.
        """
        env_t0 = time.perf_counter()

        satellite = self.satellite
        est_satellite = self.est_satellite
        controller = self.controller
        estimator = self.estimator
        orbit_estimator = self.orbit_estimator

        os_k._skip_jacobians = self._skip_jac
        os_kp1._skip_jacobians = self._skip_jac

        # Step length in seconds derived from the two environment epochs.
        dt = float((os_kp1.J2000 - os_k.J2000) * TimeConstants.cent2sec)

        x = self.x
        u = self.u

        y = satellite.sensor_readings(x=x, os=os_k)
        y_clean = satellite.noiseless_sensor_readings(x=x, os=os_k)

        if orbit_estimator is not None:
            gps = satellite.GPS_readings(x=x, os=os_k)
            os_hat = orbit_estimator.update(GPS_measurements=gps, J2000=J2000_k)
            os_for_gnc = os_hat if os_hat is not None else os_k
        else:
            os_hat = None
            os_for_gnc = os_k
        self.os_hat = os_hat

        if estimator is not None:
            self.x_hat = estimator.update(u=u, sensors=y, os=os_for_gnc)
            x_for_ctrl = self.x_hat
        else:
            x_for_ctrl = x

        active_goal = self.goal_list.get_active_goal(J2000_k, time_units="centuries")

        if controller is not None:
            u = controller.find_u(
                x_hat=x_for_ctrl,
                sens=y,
                est_sat=est_satellite,
                os_hat=os_for_gnc,
                goal=active_goal,
            )
        else:
            u = np.zeros(satellite.control_len)
        self.u = u

        env_local_time_s = time.perf_counter() - env_t0

        dyn_t0 = time.perf_counter()
        out = solve_ivp(
            fun=satellite.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os_k, os_kp1),
            rtol=1e-7,
            atol=1e-7,
        )
        dynamics_time_s = time.perf_counter() - dyn_t0
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        self.x = x

        target, w_target = active_goal.to_ref(os_for_gnc)

        boresight_vec = None
        try:
            boresight_vec = est_satellite.get_boresight(active_goal.boresight_name)
        except (AttributeError, KeyError, ValueError, TypeError):
            pass

        est_act_bias_snapshot, est_sens_bias_snapshot = self._estimated_bias_snapshots()

        self.results.record(
            k=k,
            time_J2000=J2000_k,
            time_s=k * dt,
            os=os_k,
            est_os=os_hat,
            os_cov=(getattr(getattr(orbit_estimator, "os_hat", None), "P", None)
                    if orbit_estimator is not None else None),
            state=x,
            est_state=self.x_hat,
            state_cov=(getattr(getattr(estimator, "x_hat", None), "cov", None)
                       if estimator is not None else None),
            actuator_bias=(
                np.array([np.atleast_1d(act.bias.bias) for act in satellite.actuators], dtype=object)
                if getattr(satellite, "actuators", None) else None
            ),
            sensor_bias=(
                np.array([np.atleast_1d(sens.bias.bias) for sens in satellite.sensors], dtype=object)
                if getattr(satellite, "sensors", None) else None
            ),
            est_actuator_bias=est_act_bias_snapshot,
            est_sensor_bias=est_sens_bias_snapshot,
            target=target,
            w_target=w_target,
            boresight=boresight_vec,
            clean_sensor=y_clean,
            sensor=y,
            control=u,
            control_rpc_time=getattr(controller, "last_roundtrip_s", None) if controller is not None else None,
            control_rpc_server_time=getattr(controller, "last_server_s", None) if controller is not None else None,
            env_local_time=env_local_time_s,
            dynamics_time=dynamics_time_s,
        )

        return x

    def _estimated_bias_snapshots(self):
        r"""
        Extract per-actuator / per-sensor estimated-bias snapshots from ``x_hat``.

        Mirrors the bias-slicing logic of the historical single-satellite loop;
        returns ``(est_act_bias_snapshot, est_sens_bias_snapshot)`` (each may be
        ``None``).
        """
        estimator = self.estimator
        est_satellite = self.est_satellite
        satellite = self.satellite
        x_hat = self.x_hat

        if not (estimator is not None and x_hat is not None and est_satellite is not None):
            return None, None

        n_rw = getattr(est_satellite, "number_RW", 0)
        n_ab = getattr(est_satellite, "act_bias_len", 0)
        n_sb = getattr(est_satellite, "att_sens_bias_len", 0)

        base = 7 + int(n_rw)
        ab0, ab1 = base, base + int(n_ab)
        sb0, sb1 = ab1, ab1 + int(n_sb)

        if len(x_hat) < sb1:
            return None, None

        b_act_hat = np.asarray(x_hat[ab0:ab1], dtype=float).reshape(-1)
        b_sens_hat = np.asarray(x_hat[sb0:sb1], dtype=float).reshape(-1)

        # Actuator biases
        act_parts = []
        ai = 0
        if getattr(satellite, "actuators", None):
            for act in satellite.actuators:
                if hasattr(act, "bias") and bool(act.bias) and ai + int(np.atleast_1d(act.bias.bias).size) <= b_act_hat.size:
                    dim = int(np.atleast_1d(act.bias.bias).size)
                    act_parts.append(b_act_hat[ai:ai + dim].reshape(dim, 1) if dim == 1 else b_act_hat[ai:ai + dim])
                    ai += dim
                else:
                    act_parts.append(None)
        if len(act_parts) == 0:
            est_act_bias_snapshot = None
        elif ai != b_act_hat.size:
            est_act_bias_snapshot = np.array([b_act_hat.copy()], dtype=object)
        else:
            est_act_bias_snapshot = np.array(act_parts, dtype=object)

        # Sensor biases
        sens_parts = []
        si = 0
        if getattr(satellite, "sensors", None):
            for sens in satellite.sensors:
                if hasattr(sens, "bias") and bool(sens.bias) and si + int(np.atleast_1d(sens.bias.bias).size) <= b_sens_hat.size:
                    dim = int(np.atleast_1d(sens.bias.bias).size)
                    sens_parts.append(b_sens_hat[si:si + dim].reshape(dim, 1) if dim == 1 else b_sens_hat[si:si + dim])
                    si += dim
                else:
                    sens_parts.append(None)
        if len(sens_parts) == 0:
            est_sens_bias_snapshot = None
        elif si != b_sens_hat.size:
            est_sens_bias_snapshot = np.array([b_sens_hat.copy()], dtype=object)
        else:
            est_sens_bias_snapshot = np.array(sens_parts, dtype=object)

        return est_act_bias_snapshot, est_sens_bias_snapshot
