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
    def __init__(self, est_sat: EstimatedSatellite, gain: float) -> None:
        self.gain = gain
        
        # Sensor Reading Matrix
        self.M_read = self._build_sensor_matrix(attitude_sensors=est_sat.attitude_sensors)

        # Actuation Matrix
        self.M_act = self._build_actuation_matrix(actuators=est_sat.actuators)

        # State Storage for derivative
        self.prev_B = np.zeros(3)
        self.prev_time = None
        self.initialized = False

        # Storage of max torque limits
        self.max_torque = np.array([mtq.u_max for mtq in est_sat.actuators if isinstance(mtq, MTQ)])

    def _build_sensor_matrix(self, attitude_sensors: List[Sensor]) -> np.ndarray:
        mtm_rows = []
        mtm_indices = []

        curr_idx = 0

        for sensor in attitude_sensors:
            length = sensor.output_length

            if isinstance(sensor, MTM):
                # Ensure axis is a (n_axes x 3) matrix:
                # - single-axis MTM: axis.shape == (3,) -> becomes (1, 3)
                # - tri-axis MTM: axis.shape == (3, 3) -> stays (3, 3)
                axis = np.asarray(sensor.axis, dtype=float)
                axis = axis.reshape(-1, 3)

                num_axes = axis.shape[0]

                # Each row of `axis` is a direction vector in R^3
                for r in range(num_axes):
                    mtm_rows.append(axis[r])
                    mtm_indices.append(curr_idx + r)

            curr_idx += length

        if not mtm_rows:
            raise ValueError("BDot requires at least one MTM sensor.")

        # Stack rows into H (shape: n_total_axes x 3)
        H = np.vstack(mtm_rows)

        # Pseudoinverse: maps stacked measurements -> B (3 x n_total_axes)
        H_inv = np.linalg.pinv(H)

        # Full readout matrix mapping all sensor outputs y -> B
        M_read = np.zeros((3, curr_idx))
        for i, sens_idx in enumerate(mtm_indices):
            M_read[:, sens_idx] = H_inv[:, i]

        return M_read
    
    def _build_actuation_matrix(self, actuators: List[Actuator]) -> np.ndarray:
        mtq_cols = []
        mtq_indices = []

        # So far, all actuators just have one input
        total_act_len = len(actuators)
        curr_act_idx = 0

        for act in actuators:
            if isinstance(act, MTQ):
                # Ensure axis is (3, n_cols):
                # - if axis is (3,), this becomes (3, 1)
                # - if axis is already (3, n), it stays that way
                axis = np.asarray(act.axis, dtype=float)
                axis = axis.reshape(3, -1)

                num_cols = axis.shape[1]

                for c in range(num_cols):
                    mtq_cols.append(axis[:, c])
                    mtq_indices.append(curr_act_idx + c)

            curr_act_idx += 1

        if not mtq_cols:
            raise ValueError("BDot requires at least one MTQ actuator.")

        # A maps actuator inputs -> magnetic moment (3 x n_cols)
        A = np.column_stack(mtq_cols)
        A_inv = np.linalg.pinv(A)

        # Map desired dipole moment (3,) -> actuator commands (total_act_len,)
        M_act = np.zeros((total_act_len, 3))
        for i, act_idx in enumerate(mtq_indices):
            if act_idx >= total_act_len:
                # Helpful debug if someone later adds multi-input MTQs
                raise ValueError(
                    "MTQ axis defines more inputs than there are actuator command channels. "
                    "Update total_act_len / indexing to support multi-input actuators."
                )
            M_act[act_idx, :] = A_inv[i, :]

        return M_act


    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State) -> np.ndarray:
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

        return limit(u=u_cmd, umax=self.max_torque)