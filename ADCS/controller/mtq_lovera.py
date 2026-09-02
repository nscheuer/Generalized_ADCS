__all__ = ["MTQ_Lovera"]

import numpy as np

from ADCS.state import EstimatorState, State

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat

class MTQ_Lovera(Controller):
    r"""
    Magnetic proportional–derivative attitude controller using magnetorquers only,
    based on the global stabilization law of Lovera and Astolfi [1]_.

    This controller implements the magnetic attitude control strategy presented in

    M. Lovera and A. Astolfi,
    Global Magnetic Attitude Control of Inertially Pointing Spacecraft,
    Journal of Guidance, Control, and Dynamics, Vol. 28, No. 5, 2005, pp. 1065–1072.
    Original paper: `doi:10.2514/1.11844 <https://doi.org/10.2514/1.11844>`__

    The control law achieves global asymptotic stabilization of a constant inertial
    attitude using only magnetic torquers by exploiting the time-varying nature of
    the geomagnetic field along the orbit.

    Rotational Dynamics

    The spacecraft rotational dynamics are modeled as

    .. math::

        J \dot{\boldsymbol{\omega}}
        =
        -\boldsymbol{\omega} \times (J \boldsymbol{\omega} + \boldsymbol{h}_{rw})
        + \boldsymbol{\tau}

    where

    +---------------------+----------------------------------------------------------+
    | Symbol                      | Description                                      |
    +=====================+==========================================================+
    | :math:`\boldsymbol{\omega}` | Body angular velocity                            |
    +---------------------+----------------------------------------------------------+
    | :math:`J`                   | Spacecraft inertia matrix                        |
    +---------------------+----------------------------------------------------------+
    | :math:`\boldsymbol{h}_{rw}` | Total reaction wheel angular momentum            |
    +---------------------+----------------------------------------------------------+
    | :math:`\boldsymbol{\tau}`   | Control torque                                   |
    +---------------------+----------------------------------------------------------+

    Magnetic Torque Model
    ---------------------

    Magnetorquers generate torque according to

    .. math::

        \boldsymbol{\tau} = \boldsymbol{m} \times \boldsymbol{B}

    where :math:`\boldsymbol{m}` is the commanded magnetic dipole moment and
    :math:`\boldsymbol{B}` is the geomagnetic field expressed in the body frame.

    Attitude and Angular Velocity Errors
    ------------------------------------

    Let q be the current attitude quaternion and q_d the desired inertial
    reference quaternion. The attitude error vector is defined as

    .. math::

        \boldsymbol{e}_q = \mathrm{vec}(q_d^{-1} \otimes q)

    The angular velocity error is defined as

    .. math::

        \boldsymbol{e}_{\omega}
        =
        \boldsymbol{\omega}
        -
        R_b^i(q)^\top \boldsymbol{\omega}_d

    where :math:`R_b^i(q)` is the body-to-inertial rotation matrix and
    :math:`\boldsymbol{\omega}_d` is the desired inertial angular rate.

    PD Control with Gyroscopic Compensation
    ---------------------------------------

    The desired control torque is chosen as

    .. math::

        \boldsymbol{\tau}_{des}
        =
        -\varepsilon^2 k_p \boldsymbol{e}_q
        -\varepsilon k_d \boldsymbol{e}_{\omega}
        +
        \boldsymbol{\omega} \times (J \boldsymbol{\omega} + \boldsymbol{h}_{rw})

    where :math:`k_p` and :math:`k_d` are positive scalar gains and :math:`\varepsilon` is a small
    positive parameter separating time scales.

    Magnetic Dipole Allocation
    --------------------------

    Since magnetorquers cannot generate torque parallel to the geomagnetic field,
    the commanded magnetic dipole is computed as

    .. math::

        \boldsymbol{m}^*
        =
        \frac{\boldsymbol{B} \times \boldsymbol{\tau}_{des}}{\|\boldsymbol{B}\|^2}

    which yields

    .. math::

        \boldsymbol{m}^* \times \boldsymbol{B}
        =
        \Pi_{\boldsymbol{B}^\perp}(\boldsymbol{\tau}_{des})

    If the magnetic field magnitude falls below a numerical threshold, the dipole
    command is set to zero.

    Actuator Saturation
    -------------------

    Magnetorquer dipole limits are enforced by uniformly scaling the commanded
    dipole:

    .. math::

        \boldsymbol{m} = \alpha \boldsymbol{m}^*,
        \quad
        \alpha =
        \min\left(
            1,
            \min_i \frac{m_{i,\max}}{|m_i^*|}
        \right)

    This preserves the dipole direction and therefore the resulting torque
    direction.

    References
    ----------

    .. [1] M. Lovera and A. Astolfi,
       Global Magnetic Attitude Control of Inertially Pointing Spacecraft,
       Journal of Guidance, Control, and Dynamics,
       Vol. 28, No. 5, 2005, pp. 1065–1072.
       Original paper: `doi:10.2514/1.11844 <https://doi.org/10.2514/1.11844>`__

    """
    def __init__(self, est_sat: Satellite, p_gain: float, d_gain: float, eps: float) -> None:
        r"""
        Initializes the Lovera–Astolfi magnetic attitude controller.

        This constructor stores the controller gains, constructs the magnetic
        field reconstruction matrix using onboard magnetometers, and extracts
        the maximum allowable magnetic dipole moments from the magnetorquer
        actuators.

        :param est_sat: Estimated satellite object containing sensors and actuators
        :type est_sat: ~ADCS.satellite_hardware.satellite.satellite.Satellite
        :param p_gain: Proportional gain k_p for attitude error feedback
        :type p_gain: float
        :param d_gain: Derivative gain k_d for angular velocity error feedback
        :type d_gain: float
        :param eps: Small positive time-scale separation parameter \varepsilon
        :type eps: float
        :return: None
        :rtype: None

        """
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.eps = eps

        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors+est_sat.rw_actuators, sensor_type=MTM)

        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        
    def find_u(self, x_hat: State | EstimatorState, sens: np.ndarray, est_sat: Satellite, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        r"""
        Computes magnetorquer actuator commands using the Lovera–Astolfi control law.

        This method evaluates the magnetic PD attitude stabilization controller
        using the current state estimate, sensor measurements, and mission goal.
        Reaction wheel angular momentum is included in the gyroscopic compensation
        term if present.

        The output command vector contains nonzero entries only for magnetorquer
        actuators; all other actuator commands are set to zero.

        :param x_hat: Estimated state vector containing angular velocity and attitude
        :type x_hat: ADCS.state.State | ADCS.state.EstimatorState
        :param sens: Raw sensor measurement vector
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite object providing hardware properties
        :type est_sat: ~ADCS.satellite_hardware.satellite.satellite.Satellite
        :param os_hat: Estimated orbital state
        :type os_hat: ~ADCS.orbits.orbital_state.Orbital_State
        :param goal: Optional mission goal definition
        :type goal: ~ADCS.CONOPS.goals.Goal or None
        :return: Actuator command vector with magnetorquer dipole commands
        :rtype: numpy.ndarray

        """
        if goal is None:
            goal = No_Goal()

        w = x_hat.w
        q = x_hat.q

        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        h_rw_states = self.reaction_wheel_momentum_states(
            x_hat,
            n_rw,
            context=f"{type(self).__name__}.find_u",
        )

        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        boresight = est_sat.get_boresight(goal.boresight_name)
        q_err = goal.error(q=q, body_boresight=boresight, os0=os_hat)
        w_err = w - w_ref_body
        
        tau_pd = -(self.eps**2*self.p_gain*q_err + self.eps*self.d_gain*w_err)

        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1
        
        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)

        tau_des = tau_pd + tau_gyro

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
