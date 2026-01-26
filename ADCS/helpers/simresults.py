__all__ = ["SimulationResults"]

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State

@dataclass
class SimulationResults:
    satellite: Satellite
    est_satellite: Optional[EstimatedSatellite] = None

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
    eci_target_hist: Optional[np.ndarray] = None
    w_target_hist: Optional[np.ndarray] = None

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
        eci_target=None,
        w_target=None,
        clean_sensor=None,
        sensor=None,
        control=None,
    ):
        if time_J2000 is not None:
            if self.time_J2000 is None:
                self.time_J2000 = []
            self.time_J2000.append(time_J2000)

        if time_s is not None:
            if self.time_s is None:
                self.time_s = []
            self.time_s.append(time_s)

        if os is not None:
            if self.os_hist is None:
                self.os_hist = []
            self.os_hist.append(os)

        if est_os is not None:
            if self.est_os_hist is None:
                self.est_os_hist = []
            self.est_os_hist.append(est_os)

        if os_cov is not None:
            if self.os_cov_hist is None:
                self.os_cov_hist = []
            self.os_cov_hist.append(os_cov)

        if state is not None:
            if self.state_hist is None:
                self.state_hist = []
            self.state_hist.append(np.asarray(state).copy())

        if est_state is not None:
            if self.est_state_hist is None:
                self.est_state_hist = []
            self.est_state_hist.append(np.asarray(est_state).copy())

        if state_cov is not None:
            if self.state_cov_hist is None:
                self.state_cov_hist = []
            self.state_cov_hist.append(state_cov)

        if sensor_bias is not None:
            if self.sensor_bias is None:
                self.sensor_bias = []
            self.sensor_bias.append(sensor_bias)

        if est_sensor_bias is not None:
            if self.est_sensor_bias is None:
                self.est_sensor_bias = []
            self.est_sensor_bias.append(est_sensor_bias)

        if actuator_bias is not None:
            if self.actuator_bias is None:
                self.actuator_bias = []
            self.actuator_bias.append(actuator_bias)

        if est_actuator_bias is not None:
            if self.est_actuator_bias is None:
                self.est_actuator_bias = []
            self.est_actuator_bias.append(est_actuator_bias)

        if eci_target is not None:
            if self.eci_target_hist is None:
                self.eci_target_hist = []
            self.eci_target_hist.append(np.asarray(eci_target))

        if w_target is not None:
            if self.w_target_hist is None:
                self.w_target_hist = []
            self.w_target_hist.append(np.asarray(w_target))

        if clean_sensor is not None:
            if self.clean_sensor_hist is None:
                self.clean_sensor_hist = []
            self.clean_sensor_hist.append(np.asarray(clean_sensor))

        if sensor is not None:
            if self.sensor_hist is None:
                self.sensor_hist = []
            self.sensor_hist.append(np.asarray(sensor))

        if control is not None:
            if self.control_hist is None:
                self.control_hist = []
            self.control_hist.append(np.asarray(control))
