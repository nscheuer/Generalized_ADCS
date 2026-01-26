__all__ = ["SimulationResults"]

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from ADCS.orbits.orbital_state import Orbital_State

@dataclass
class SimulationResults:
    time_J2000: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None

    os_hist: Optional[List[Orbital_State]] = None
    est_os_hist: Optional[List[Orbital_State]] = None
    os_cov_hist: Optional[List[np.ndarray]] = None

    state_hist: Optional[np.ndarray] = None
    est_state_hist: Optional[np.ndarray] = None
    state_cov_hist: Optional[List[np.ndarray]] = None

    sensor_bias: Optional[np.ndarray] = None
    est_sensor_bias: Optional[np.ndarray] = None

    actuator_bias: Optional[np.ndarray] = None
    est_actuator_bias: Optional[np.ndarray] = None

    clean_sensor_hist: Optional[np.ndarray] = None
    sensor_hist: Optional[np.ndarray] = None

    control_hist: Optional[np.ndarray] = None

    def record(
        self,
        *,
        k: int,
        time_J2000=None,
        time_s=None,
        os=None,
        est_os=None,
        os_cov=None,
        state=None,
        est_state=None,
        state_cov=None,
        sensor_bias=None,
        est_sensor_bias=None,
        actuator_bias=None,
        est_actuator_bias=None,
        clean_sensor=None,
        sensor=None,
        control=None,
    ):
        if self.time_J2000 is not None and time_J2000 is not None:
            self.time_J2000[k] = time_J2000

        if self.time_s is not None and time_s is not None:
            self.time_s[k] = time_s

        if self.os_hist is not None and os is not None:
            self.os_hist.append(os)

        if self.est_os_hist is not None and est_os is not None:
            self.est_os_hist.append(est_os)

        if self.os_cov_hist is not None and os_cov is not None:
            self.os_cov_hist.append(os_cov)

        if self.state_hist is not None and state is not None:
            self.state_hist[k] = state

        if self.est_state_hist is not None and est_state is not None:
            self.est_state_hist[k] = est_state

        if self.state_cov_hist is not None and state_cov is not None:
            self.state_cov_hist.append(state_cov)

        if self.sensor_bias is not None and sensor_bias is not None:
            self.sensor_bias[k] = sensor_bias

        if self.est_sensor_bias is not None and est_sensor_bias is not None:
            self.est_sensor_bias[k] = est_sensor_bias

        if self.actuator_bias is not None and actuator_bias is not None:
            self.actuator_bias[k] = actuator_bias

        if self.est_actuator_bias is not None and est_actuator_bias is not None:
            self.est_actuator_bias[k] = est_actuator_bias

        if self.clean_sensor_hist is not None and clean_sensor is not None:
            self.clean_sensor_hist[k] = clean_sensor

        if self.sensor_hist is not None and sensor is not None:
            self.sensor_hist[k] = sensor

        if self.control_hist is not None and control is not None:
            self.control_hist[k] = control
