__all__ = ["BDot"]

import numpy as np
from typing import List

from ADCS.controller.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Sensor, MTM
from ADCS.satellite_hardware.actuators import Actuator, MTQ
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import limit

class BDot(Controller):
    r"""
    Implements the B-Dot detumbling controller.

    The B-Dot algorithm is a magnetic control law used primarily for detumbling a satellite 
    (reducing angular velocity) after deployment. It relies on the interaction between 
    magnetic actuators (magnetorquers) and the Earth's magnetic field.

    **Control Law**

    The controller calculates a desired magnetic dipole moment :math:`\mathbf{m}` based on the 
    rate of change of the measured magnetic field body vector :math:`\mathbf{B}`.

    .. math::

        \mathbf{m}_{req} = -K \dot{\mathbf{B}}

    Where:
        - :math:`\mathbf{m}_{req}` is the requested magnetic dipole moment (:math:`Am^2`).
        - :math:`K` is the control gain (positive scalar).
        - :math:`\dot{\mathbf{B}}` is the time derivative of the magnetic field vector in the body frame.

    **Physical Principle**

    The torque generated is :math:`\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}`. 
    Substituting the control law:

    .. math::

        \boldsymbol{\tau} = -K (\dot{\mathbf{B}} \times \mathbf{B})

    For a tumbling satellite, the change in the magnetic field vector is dominated by the 
    satellite's rotation :math:`\boldsymbol{\omega}` rather than the orbital change of the 
    magnetic field. Therefore, :math:`\dot{\mathbf{B}} \approx \boldsymbol{\omega} \times \mathbf{B}`. 
    This results in a torque that opposes the angular velocity perpendicular to the magnetic field, 
    dissipating kinetic energy.

    **Hardware Prerequisites**

    This controller requires specific hardware interfaces defined in the :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`:

    1.  **Sensors:** At least one :class:`~ADCS.satellite_hardware.sensors.MTM` (Magnetometer) to measure :math:`\mathbf{B}`.
    2.  **Actuators:** At least one :class:`~ADCS.satellite_hardware.actuators.MTQ` (Magnetorquer) to generate :math:`\mathbf{m}`.

    **Matrix Mapping**

    This class automatically builds mapping matrices using the Moore-Penrose pseudoinverse to handle 
    configurations with redundant sensors or coupled actuator axes.

    :param est_sat: The estimated satellite object containing hardware definitions.
    :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
    :param gain: The proportional gain :math:`K` applied to the B-field derivative.
    :type gain: float
    """
    def __init__(self, est_sat: EstimatedSatellite, gain: float) -> None:
        self.gain = gain
        
        # Sensor Reading Matrix
        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(sensors=est_sat.attitude_sensors, sensor_type=MTM)

        # Actuation Matrix
        self.M_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(actuators=est_sat.actuators, actuator_type=MTQ)

        # State Storage for derivative
        self.prev_B = np.zeros(3)
        self.prev_time = None
        self.initialized = False

        # Storage of max torque limits
        self.max_torque = self.find_max_torque(actuators=est_sat.actuators, actuator_type=MTQ)

        self.n_actuators = len(est_sat.actuators)


    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State) -> np.ndarray:
        r"""
        Computes the control signal based on the B-Dot law.

        This method performs the following steps:
        
        1.  **Reconstruct B-field:** Transforms raw sensor data ``sens`` into the magnetic field vector 
            :math:`\mathbf{B}_{curr}` using the pre-computed :math:`M_{read}`.
        2.  **Calculate Derivative:** Computes the discrete time derivative :math:`\dot{\mathbf{B}}` 
            using a backward difference approximation:

            .. math::
            
                \dot{\mathbf{B}} \approx \frac{\mathbf{B}_{curr} - \mathbf{B}_{prev}}{\Delta t}

        3.  **Compute Dipole:** Applies the gain :math:`\mathbf{m}_{desired} = -K \dot{\mathbf{B}}`.
        4.  **Map to Actuators:** Converts the desired dipole to specific actuator commands via :math:`M_{act}`.
        5.  **Saturate:** Limits the output to :math:`u_{max}` of the specific magnetorquers.

        :param x_hat: The current state estimate vector (unused in B-Dot, relies on raw sensors).
        :type x_hat: np.ndarray
        :param sens: The raw sensor measurements array.
        :type sens: np.ndarray
        :param est_sat: The estimated satellite object (unused in loop, required by interface).
        :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
        :param os_hat: The current orbital state estimate (used for J2000 timestamps).
        :type os_hat: ~ADCS.orbits.orbital_state.Orbital_State
        :return: The computed control signal :math:`\mathbf{u}` (e.g., duty cycles or voltages).
        :rtype: np.ndarray
        """
        y = np.asarray(sens).reshape(-1)
        B_curr = self.M_read @ y

        t_curr = os_hat.J2000

        if not self.initialized or self.prev_time is None:
            B_dot = np.zeros(3)
            self.initialized = True
        else:
            dt = t_curr - self.prev_time
            dt*=TimeConstants.cent2sec
            if dt <= 1e-9:
                B_dot = np.zeros(3)
            else:
                B_dot = (B_curr - self.prev_B)/dt

        self.prev_B = B_curr
        self.prev_time = t_curr

        m_desired = -self.gain * B_dot
        u_cmd = self.M_act @ m_desired

        u_cmd = limit(u=u_cmd, umax=self.max_torque)
        u_total = np.zeros(self.n_actuators)
        u_total[self.mtq_indices] = u_cmd

        return u_total