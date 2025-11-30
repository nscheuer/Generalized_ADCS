import numpy as np
from typing import Dict

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.estimators.orbit_estimators import Orbit_Estimator
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.controller import Controller
from ADCS.CONOPS.rulebook import Rulebook

class ADCS():
    def __init__(self, est_sat: EstimatedSatellite, orbit_estimators: Dict[str, Orbit_Estimator], attitude_estimators: Dict[str, Attitude_Estimator], controllers: Dict[str, Controller], rulebook: Rulebook, trajectory_planner = None) -> None:
        self.est_sat = est_sat
        self.orbit_estimators = orbit_estimators
        self.attitude_estimators = attitude_estimators
        self.controllers = controllers
        self.rulebook = rulebook
        self.trajectory_planner = trajectory_planner

        self.orbit_estimator: Orbit_Estimator = self.orbit_estimators[0]


    def update(self, t: float, sensor_readings: np.ndarray)
