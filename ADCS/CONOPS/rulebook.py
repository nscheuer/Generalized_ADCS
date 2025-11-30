import numpy as np
from typing import List, Callable, Optional

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.estimators.attitude_estimators import Attitude_Estimator
from ADCS.controller import Controller

RuleFunction = Callable[[float, np.ndarray, EstimatedSatellite, Orbital_State], bool]

class Rule():
    def __init__(self, logic_function: RuleFunction | bool, mode: str):
        if logic_function is True:
            self.logic = lambda *args: True
        elif logic_function is False:
            self.logic = lambda *args: False
        else:
            self.logic = logic_function
        self.mode = mode

    def check(self, t: float, x_hat: np.ndarray, est_sat: EstimatedSatellite, est_os: Orbital_State) -> bool:
        return self.logic(t, x_hat, est_sat, est_os)
    
    def __repr__(self):
        return f"<Rule: {self.mode}>"


class Rulebook():
    def __init__(self, attitude_estimator_rules: List[Rule], orbital_estimator_rules: List[Rule], controller_rules: List[Rule]):
        self.attitude_estimator_rules = attitude_estimator_rules
        self.orbital_estimator_rules = orbital_estimator_rules
        self.controller_rules = controller_rules
    
    #TODO: Add hysteresis

    def _get_first_true(self, rules: List[Rule], x_hat: np.ndarray, est_sat: EstimatedSatellite, est_os: Orbital_State) -> str:
        for rule in rules:
            if rule.check(x_hat, est_sat, est_os):
                return rule.mode 
        # Otherwise default to safe state  
        return rules[0].mode

    def select_attitude_estimator(self, t: float, x_hat: np.ndarray, est_sat: EstimatedSatellite, est_os: Orbital_State) -> str:
        return self._get_first_true(self.attitude_estimator_rules, x_hat, est_sat, est_os)
    
    def select_orbital_estimator(self, t: float, x_hat: np.ndarray, est_sat: EstimatedSatellite, est_os: Orbital_State) -> str:
        return self._get_first_true(self.orbital_estimator_rules, x_hat, est_sat, est_os)
    
    def select_controller(self, t: float, x_hat: np.ndarray, est_sat: EstimatedSatellite, est_os: Orbital_State) -> str:
        return self._get_first_true(self.controller_rules, x_hat, est_sat, est_os) 