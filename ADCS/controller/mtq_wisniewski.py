__all__ = ["MTQ_Wisniewski"]

import numpy as np

from ADCS.state import EstimatorState, State

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, Wmat

class MTQ_Wisniewski(Controller):
    r"""
    Sliding mode magnetic attitude controller based on Wisniewski (1998).

    This controller implements a sliding mode control law for spacecraft
    attitude regulation using magnetic torquers as the sole actuation
    mechanism. The formulation follows the work of R. Wisniewski and is
    designed to handle the nonlinear, time-varying, and underactuated
    nature of magnetic attitude control.

    The controller defines a sliding surface in terms of angular velocity
    error and attitude error. Control action is chosen such that system
    trajectories are driven onto this surface and remain there, ensuring
    robustness to modeling uncertainties and disturbances.

    Attitude and angular velocity errors are defined as:

    .. math::

        \boldsymbol{\omega}_{\mathrm{err}} =
        \boldsymbol{\omega} - \boldsymbol{\omega}_{\mathrm{ref}}

    .. math::

        \boldsymbol{q}_{\mathrm{err}} \in \mathbb{R}^3

    The sliding surface is constructed as:

    .. math::

        \boldsymbol{s} =
        J \boldsymbol{\omega}_{\mathrm{err}} +
        \Lambda_q \boldsymbol{q}_{\mathrm{err}}

    where :math:`J` is the spacecraft inertia matrix and
    :math:`\Lambda_q` is a positive definite gain matrix.

    A Lyapunov candidate function is chosen as:

    .. math::

        V = \frac{1}{2} \boldsymbol{s}^\mathsf{T} \boldsymbol{s}

    Enforcing :math:`\dot{V} < 0` leads to the desired control torque:

    .. math::

        \boldsymbol{\tau}_{\mathrm{des}} =
        \boldsymbol{\omega} \times (J\boldsymbol{\omega} + \boldsymbol{h}_{\mathrm{rw}})
        + J (\boldsymbol{\omega} \times \boldsymbol{\omega}_{\mathrm{err}})
        - \Lambda_q \dot{\boldsymbol{q}}_{\mathrm{err}}
        - \Lambda_s \boldsymbol{s}

    Since magnetic actuation can only generate torque orthogonal to the
    local magnetic field, the desired torque is mapped to a magnetic
    dipole moment using:

    .. math::

        \boldsymbol{m} =
        \frac{\boldsymbol{B} \times \boldsymbol{\tau}_{\mathrm{des}}}
        {\lVert \boldsymbol{B} \rVert^2}

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Estimated satellite model providing inertia, actuator geometry,
        and sensor configuration.
    lambda_s : numpy.ndarray
        Positive definite sliding surface gain matrix.
    lambda_q : numpy.ndarray
        Positive definite attitude error gain matrix.

    Returns
    -------
    None

    References
    ----------
    R. Wisniewski,
    Sliding Mode Attitude Control for Magnetic Actuated Satellite,
    IFAC Proceedings Volumes, Vol. 31, No. 18, 1998.

    """
    def __init__(self, est_sat: EstimatedSatellite, lambda_s: np.ndarray, lambda_q: np.ndarray) -> None:
        r"""
        Initialize the Wisniewski sliding mode magnetic controller.

        This method stores the sliding surface gain matrices, constructs
        the magnetic field reconstruction mapping from magnetometer
        sensors, and extracts magnetic torquer actuation limits from the
        satellite model.

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite object containing sensors, actuators,
            and inertia properties.
        lambda_s : numpy.ndarray
            Sliding surface convergence gain matrix.
        lambda_q : numpy.ndarray
            Attitude error gain matrix.

        Returns
        -------
        None

        """
        self.lambda_s = lambda_s
        self.lambda_q = lambda_q

        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors+est_sat.rw_actuators, sensor_type=MTM)

        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        
    def find_u(self, x_hat: EstimatorState, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        r"""
        Compute magnetic torquer command vector using sliding mode control.

        This method evaluates the estimated spacecraft state and sensor
        measurements to compute a magnetic dipole command that enforces
        the sliding mode dynamics. Reaction wheel momentum is included
        in the gyroscopic compensation term when available.

        The sliding surface is computed as:

        .. math::

            \boldsymbol{s} =
            J \boldsymbol{\omega}_{\mathrm{err}} +
            \Lambda_q \boldsymbol{q}_{\mathrm{err}}

        The desired control torque is formed as:

        .. math::

            \boldsymbol{\tau}_{\mathrm{des}} =
            \boldsymbol{\omega} \times (J\boldsymbol{\omega} + \boldsymbol{h}_{\mathrm{rw}})
            + J (\boldsymbol{\omega} \times \boldsymbol{\omega}_{\mathrm{err}})
            - \Lambda_q \dot{\boldsymbol{q}}_{\mathrm{err}}
            - \Lambda_s \boldsymbol{s}

        Magnetic dipole allocation is performed using the cross-product
        projection law:

        .. math::

            \boldsymbol{u}_{\mathrm{mtq}} =
            \frac{\boldsymbol{B} \times \boldsymbol{\tau}_{\mathrm{des}}}
            {\lVert \boldsymbol{B} \rVert^2}

        Actuator saturation is enforced by uniform scaling to respect
        maximum dipole moment limits.

        Parameters
        ----------
        x_hat : numpy.ndarray
            Estimated spacecraft state vector containing angular rates,
            attitude quaternion, and optional reaction wheel momentum states.
        sens : numpy.ndarray
            Raw magnetometer sensor measurements.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model providing actuator configuration and inertia.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state providing geomagnetic field information.
        goal : :class:`~ADCS.CONOPS.goals.Goal`, optional
            Attitude guidance goal used to compute reference attitude and
            angular velocity.

        Returns
        -------
        numpy.ndarray
            Actuator command vector containing magnetic torquer commands
            ordered according to ``est_sat.actuators``.

        """
        if goal is None:
            goal = No_Goal()

        w = x_hat.w
        q = x_hat.q

        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        if x_hat.h.size >= n_rw:
            h_rw_states = x_hat.h
        else:
            h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        boresight = est_sat.get_boresight(goal.boresight_name)
        q_err = goal.error(q=q, body_boresight=boresight, os0=os_hat)
        w_err = w - w_ref_body

        # Calculate sliding surface
        J = est_sat.J_0

        s = J @ w_err + self.lambda_q @ q_err 

        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1

        q_err_full = np.hstack(([np.sqrt(max(0.0, 1.0 - np.dot(q_err, q_err)))], q_err))
        q_err_dot = 0.5*w_err@Wmat(q_err_full).T

        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_frame = J @ np.cross(w, w_err)
        tau_q_err_dot = self.lambda_q @ q_err_dot[1:4]
        tau_sliding = self.lambda_s @ s

        tau_des = tau_gyro + tau_frame - tau_q_err_dot - tau_sliding

        sens = np.asarray(sens).reshape(-1)
        sens_clean = sens.copy()
        sens_clean[np.isnan(sens_clean)] = 0.0
        B_curr = self.M_read @ sens_clean
        B_norm_sq = np.linalg.norm(B_curr)**2

        if B_norm_sq < 1e-11:
            u_mtq_cmd = np.zeros(3)
        else:
            u_mtq_cmd = np.cross(B_curr, tau_des) / B_norm_sq

        scale = np.min(
            np.where(np.abs(u_mtq_cmd) > 0.0,
                    self.mtq_umax / np.abs(u_mtq_cmd),
                    np.inf)
        )
        u_mtq_cmd *= min(1.0, scale)

        u_out = np.zeros(len(est_sat.actuators))
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = u_mtq_cmd

        return u_out
