__all__ = ["MTQ_w_RW"]

import numpy as np
from typing import List

from ADCS.CONOPS.goals import Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Actuator, MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, normalize, skewsym, limit, norm

class MTQ_w_RW(Controller):
    r"""
    Hybrid magnetic torque and reaction wheel controller.

    This controller implements a three-axis ADCS control law which blends
    reaction wheel torque allocation with continuous momentum dumping via
    magnetic torquers. The implementation follows the core concepts of
    redundant wheel momentum management described by Hogan & Schaub [1]_,
    where wheel momentum states are actively controlled to remain bounded
    using MTQ torque authority.

    The controller performs:
    
    - quaternion-based attitude regulation or target alignment,
    - PD angular rate feedback,
    - reaction wheel momentum reconstruction from state,
    - gyroscopic coupling compensation,
    - active momentum unloading,
    - magnetic dipole allocation via pseudo-inverse torque mapping,
    - actuator saturation handling.

    The torque law can be summarized as:

    .. math::

        \tau_{\text{att}} =
        -K_p \mathbf{q}_{\text{err}}
        -K_d (\boldsymbol{\omega} - \boldsymbol{\omega}_{\text{ref}})
        + \tau_{\text{gyro}}

    where

    .. math::

        \tau_{\text{gyro}} =
        \boldsymbol{\omega} \times (J\boldsymbol{\omega} + \mathbf{h}_{rw})

    and the commanded magnetic dipole vector :math:`\mathbf{u}_{mtq}` is computed
    from a pseudo-inverse allocation of the momentum dumping torque

    .. math::

        \tau_{dump} = -K_c(\mathbf{h}_{rw} - \mathbf{h}_{tgt})

    subject to MTQ torque constraints

    .. math::

        \tau_{mag} = \mathbf{u}_{mtq} \times \mathbf{B}

    yielding the linearized form

    .. math::

        \tau_{mag} = -[B]_\times \, \mathbf{A}_{mtq} \, \mathbf{u}_{mtq}
                   = M_{mag} \, \mathbf{u}_{mtq}.

    Reaction wheels apply the remaining torque after MTQ authority:

    .. math::

        \tau_{rw} =
        \tau_{att} - \tau_{mag}.

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Estimated satellite object containing inertia, actuator geometry,
        and sensor configuration.
    p_gain : float
        Proportional gain :math:`K_p` used for quaternion error feedback.
    d_gain : float
        Derivative gain :math:`K_d` used for angular rate feedback.
    c_gain : float
        Momentum dumping gain :math:`K_c` applied to wheel momentum deviation.
    h_target : np.ndarray (shape: (3,))
        Desired body-frame reaction wheel momentum target in units of
        angular momentum. Dimensionality must match the number of RW axes.

    Raises
    ------
    ValueError
        If any dimension of ``h_target`` exceeds the maximum permitted
        reaction wheel storage, as determined by
        :attr:`~ADCS.satellite_hardware.actuators.RW.h_max`.

    Notes
    -----
    This implementation assumes:
      • Wheel momentum states begin at index 7 of ``x_hat``  
        (i.e. ``x_hat[7:]`` maps directly to ordered RW indices)  
      • Wheel axes align with internally-tracked actuator axis vectors
      • MTQs are torque-limited and can only generate torque orthogonal
        to the local magnetic field

    The pseudo-inverse allocation is robust for full tri-axial MTQ
    configurations or over-actuated cases.

    References
    ----------
    .. [1] E. A. Hogan and H. Schaub,
       *Three-Axis Attitude Control Using Redundant Reaction Wheels
       with Continuous Momentum Dumping*,
       Journal of Guidance, Control, and Dynamics, Vol. 44, No. 3, 2021.
    """
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray) -> None:
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.c_gain = c_gain

        self.M_mtm_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.sensors+est_sat.rw_actuators, sensor_type=MTM)

        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(actuators=est_sat.actuators, actuator_type=RW)
        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(actuators=est_sat.actuators, actuator_type=MTQ)
        self.A_mtq = self.build_u_to_torque_matrix_pinv(actuators=est_sat.actuators, actuator_type=MTQ)

        self.rw_max_h = np.asarray([rw.h_max for rw in est_sat.actuators if isinstance(rw, RW)])
        self.max_torque = self.find_max_torque(actuators=est_sat.actuators)

        if np.any(self.rw_max_h < h_target):
            raise ValueError("Target momentum cannot be higher than reaction wheel maximum momentum!")
        self.h_target = h_target

        self.n_actuators = len(est_sat.actuators)

    
    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        r"""
        Compute actuator commands for reaction wheels and magnetic torquers.

        This method evaluates the sensor state, angular rate, quaternion
        orientation, and reaction wheel stored momenta to provide the hybrid
        torque solution. Magnetic torquers continuously unload reaction wheel
        momentum while reaction wheels primarily provide fine torque authority.

        The spacecraft wheel momentum vector is reconstructed from the state:

        .. math::

            \mathbf{h}_{rw} =
            \sum_i h_i \, \hat{a}_i

        where ``h_i`` are the scalar momentum states ``x_hat[7:]`` and
        :math:`\hat{a}_i` are normalized RW axis vectors.

        Components of the total torque model:

        1. Attitude stabilization
           .. math::
               \tau_{\text{att}} =
               -K_p \mathbf{q}_{err} - K_d (\omega - \omega_{\text{ref}})
               + \tau_{\text{gyro}}

        2. Gyroscopic coupling from wheels and body
           .. math::
               \tau_{\text{gyro}} =
               \omega \times (J\omega + h_{rw})

        3. Momentum dumping
           .. math::
               \tau_{\text{dump}} =
               -K_c(\mathbf{h}_{rw} - \mathbf{h}_{tgt})

        4. Magnetic allocation
           .. math::
               \tau_{mag} = M_{mag} \, u_{mtq}

           where

           .. math::
               M_{mag} = -[B]_\times A_{mtq}

        Reaction wheel torque request is then

        .. math::
            \tau_{rw} = \tau_{\text{att}} - \tau_{\text{mag}}

        and actuator command vectors follow:

        .. math::
            u_{rw} = M_{rw} \, \tau_{rw}\\
            u_{mtq} = M_{mag}^+ \, \tau_{dump}

        Saturation is enforced using element-wise limiting.

        Parameters
        ----------
        x_hat : np.ndarray (shape: (10,))
            Estimated spacecraft state vector containing:
              • angular rates,
              • quaternion attitude,
              • reaction wheel momentum states.
        sens : np.ndarray
            Raw sensor readouts. Only MTM channels are used to reconstruct
            the body magnetic field using the precomputed pseudo-inverse
            mapping.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model containing inertia and actuator geometry.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state estimate providing the magnetic field :math:`B`
            and atmospheric density for drag estimation if applicable.
        goal_vector_eci : np.ndarray, optional
            Optional inertial-frame target direction. If provided, the
            controller operates in a pointing mode, forming the quaternion
            error from the boresight cross product.
        w_ref : np.ndarray, optional
            Desired angular rate vector. Defaults to zero if omitted.

        Returns
        -------
        u_total : np.ndarray
            Full actuator command vector ordered by actuator index in
            ``est_sat.actuators``. Reaction wheel torques occupy the RW
            allocation slots and MTQ current/command magnitudes occupy the
            MTQ slots.

        See Also
        --------
        :meth:`~ADCS.helpers.math_helpers.skewsym`
        :meth:`~ADCS.helpers.math_helpers.normalize`
        :meth:`~ADCS.controller.Controller.build_torque_to_u_matrix_pinv`
        :class:`~ADCS.satellite_hardware.actuators.RW`
        :class:`~ADCS.satellite_hardware.actuators.MTQ`

        Notes
        -----
        - MTQs produce no torque component parallel to :math:`\mathbf{B}`.
        - Reaction wheel torque authority is constrained by
          ``rw.max_torque`` and their energy storage ``h_max``.
        - This controller assumes RW momenta are stored in
          ``x_hat[7:]`` in the same order as RW indices.

        References
        ----------
        .. [1] E. A. Hogan and H. Schaub,
           *Three-Axis Attitude Control Using Redundant Reaction Wheels
           with Continuous Momentum Dumping*,
           Journal of Guidance, Control, and Dynamics, Vol. 44, No. 3, 2021.
        """
        w = x_hat[0:3]
        q = x_hat[3:7]

        if goal is None:
            goal = Goal()

        goal_vector_eci, w_ref_eci = goal.to_ref(os0=os_hat)

        b_body = self.M_mtm_read @ sens # Already filters out only MTM readings

        q_err_vec = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
        # q_err_vec = vector_alignment_error(q=q, eci_goal=goal_vector_eci, body_boresight=est_sat.boresight)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci
        w_err = w - w_ref_body

        # PD Control Law
        tau_att = -self.p_gain*q_err_vec - self.d_gain*w_err

        # Momentum Management
        h_vals = x_hat[7:]
        rw_axes   = np.vstack([
            np.asarray(rw.axis, float).reshape(3,)
            for rw in est_sat.actuators
            if isinstance(rw, RW)
        ])
        h_vals    = x_hat[7:]
        h_rw_body = h_vals @ rw_axes

        # Gyroscopic Compensation
        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        tau_att += tau_gyro
        
        # Momentum dumping
        delta_h = h_rw_body - self.h_target
        tau_dump = -self.c_gain * delta_h

        # Allocation
        B_skew = skewsym(b_body)
        M_mag_eff = -B_skew @ self.A_mtq # (3, N_MTQ)

        u_mtq = np.linalg.pinv(M_mag_eff) @ tau_dump
        u_mtq = limit(u=u_mtq, umax=self.max_torque) #TODO: Proper limiting of Actuators

        tau_mag_actual = M_mag_eff @ u_mtq
        tau_rw_req = tau_att - tau_mag_actual

        u_rw = self.M_rw_act @ tau_rw_req
        u_rw = limit(u=u_rw, umax=self.max_torque)

        u_total = u_mtq+u_rw

        return u_total
    




        