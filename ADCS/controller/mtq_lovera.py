__all__ = ["MTQ_Lovera"]

import numpy as np

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat

class MTQ_Lovera(Controller):
    r"""
    Magnetic PD Attitude Controller (Lovera–Astolfi)

    This controller implements the global magnetic attitude stabilization law
    proposed in:

        M. Lovera and A. Astolfi,
        *Global Magnetic Attitude Control of Inertially Pointing Spacecraft*,
        Journal of Guidance, Control, and Dynamics, Vol. 28, No. 5, 2005,
        pp. 1065–1072.

    The controller achieves global asymptotic stabilization of an inertially
    fixed attitude using **magnetorquers only**, exploiting the time-varying
    nature of the geomagnetic field along the orbit.

    ---------------------------------------------------------------------------
    Mathematical Background
    ---------------------------------------------------------------------------

    Let the spacecraft rotational dynamics be

    .. math::

        J \dot{\omega} = -\omega \times (J\omega + h_{rw}) + \tau

    where
    - :math:`\omega \in \mathbb{R}^3` is the body angular velocity,
    - :math:`J` is the inertia matrix,
    - :math:`h_{rw}` is the total reaction-wheel angular momentum (if present),
    - :math:`\tau` is the control torque.

    Magnetic actuators generate torque according to

    .. math::

        \tau = m \times B

    where
    - :math:`m \in \mathbb{R}^3` is the commanded magnetic dipole moment,
    - :math:`B \in \mathbb{R}^3` is the geomagnetic field expressed in body frame.

    ---------------------------------------------------------------------------
    Reference Tracking Errors
    ---------------------------------------------------------------------------

    Let :math:`q` be the spacecraft attitude quaternion and
    :math:`q_d` the desired inertial reference attitude.
    The attitude error vector is computed as

    .. math::

        e_q = \text{vec}(q_d^{-1} \otimes q)

    where :math:`\otimes` denotes quaternion multiplication.

    The angular velocity error is

    .. math::

        e_\omega = \omega - R_{b}^{i}(q)^T \omega_d

    where :math:`\omega_d` is the reference angular velocity expressed in ECI
    coordinates.

    ---------------------------------------------------------------------------
    PD + Gyroscopic Compensation Law
    ---------------------------------------------------------------------------

    The desired control torque is defined as

    .. math::

        \tau_{des} =
        -\varepsilon^2 k_p e_q
        -\varepsilon k_d e_\omega
        + \omega \times (J\omega + h_{rw})

    where
    - :math:`k_p`, :math:`k_d` are positive scalar gains,
    - :math:`\varepsilon > 0` is a small tuning parameter separating time scales.

    This structure ensures that the closed-loop dynamics can be cast in a
    singular perturbation framework, as shown in the cited paper.

    ---------------------------------------------------------------------------
    Magnetic Dipole Command
    ---------------------------------------------------------------------------

    Since magnetic torquers cannot generate torque parallel to the geomagnetic
    field, the commanded dipole is chosen as

    .. math::

        m^* = \frac{B \times \tau_{des}}{\|B\|^2}

    which guarantees

    .. math::

        m^* \times B = \Pi_{B^\perp}(\tau_{des})

    i.e. the projection of the desired torque onto the plane orthogonal to
    :math:`B`.

    If :math:`\|B\|` is below a numerical threshold, the dipole command is set
    to zero.

    ---------------------------------------------------------------------------
    Actuator Saturation Handling
    ---------------------------------------------------------------------------

    Each magnetorquer is subject to dipole magnitude limits

    .. math::

        |m_i| \le m_{i,\max}

    Rather than applying component-wise saturation (which would distort the
    torque direction), the dipole command is **uniformly scaled**:

    .. math::

        m = \alpha m^*, \quad
        \alpha = \min\!\left(1,
        \min_i \frac{m_{i,\max}}{|m^*_i|}
        \right)

    This preserves the direction of the magnetic dipole—and therefore the
    resulting torque direction—while guaranteeing feasibility.

    This saturation strategy is consistent with the assumptions used in the
    stability analysis of Lovera & Astolfi and is standard practice in
    flight-qualified magnetic attitude controllers.

    ---------------------------------------------------------------------------
    References
    ---------------------------------------------------------------------------

    .. [1] M. Lovera and A. Astolfi,
        *Global Magnetic Attitude Control of Inertially Pointing Spacecraft*,
        Journal of Guidance, Control, and Dynamics,
        Vol. 28, No. 5, 2005, pp. 1065–1072.
    """
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, eps: float) -> None:
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.eps = eps

        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors+est_sat.rw_actuators, sensor_type=MTM)

        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        
    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None = None) -> np.ndarray:
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        if len(x_hat) >= 7 + n_rw:
            h_rw_states = x_hat[7 : 7 + n_rw]
        else:
            h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
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

        y = np.asarray(sens).reshape(-1)
        B_curr = self.M_read @ y
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