"""Direct body-frame disturbance-torque model for augmented estimators."""

from __future__ import annotations

import numpy as np

from .disturbance import Disturbance

__all__ = ["Torque_Disturbance"]


class Torque_Disturbance(Disturbance):
    """Represent an unknown body-frame torque as three estimated parameters."""

    def __init__(self, torque, *, estimate_dist: bool = False):
        self.torque_nominal = np.asarray(torque, dtype=float).reshape(3).copy()
        self.current_torque = self.torque_nominal.copy()
        super().__init__(estimate_dist=estimate_dist, estimated_vector_length=3)

    @property
    def main_param(self):
        return self.torque_nominal.copy()

    @main_param.setter
    def main_param(self, value):
        self.torque_nominal = np.asarray(value, dtype=float).reshape(3).copy()
        self.current_torque = self.torque_nominal.copy()

    def update(self):
        self.current_torque = self.torque_nominal.copy()

    def torque(self, *args, **kwargs):
        return self.current_torque.copy()

    def torque_qjac(self, *args, **kwargs):
        return np.zeros((4, 3))

    def torque_valjac(self, *args, **kwargs):
        return np.eye(3)

    def torque_qqhess(self, *args, **kwargs):
        return np.zeros((4, 4, 3))

    def torque_qvalhess(self, *args, **kwargs):
        return np.zeros((4, 3, 3))

    def torque_valvalhess(self, *args, **kwargs):
        return np.zeros((3, 3, 3))
