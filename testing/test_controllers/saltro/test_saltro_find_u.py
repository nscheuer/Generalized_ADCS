"""SALTRO.find_u clips the tracking command to actuator limits.

These tests bypass the planner (no compiled SALTRO module needed): a stub
trajectory supplies the command and ``find_u`` is called directly.
"""

from types import SimpleNamespace

import numpy as np

from ADCS.controller.saltro.saltro_controller import SALTRO


class _StubTrajectory:
    start_time = 0.0
    end_time = 1.0

    def __init__(self, command):
        self.command = np.asarray(command, dtype=float)

    def is_valid_time(self, t):
        return self.start_time <= t <= self.end_time

    def compute_tracking_control(self, t, x_hat):
        return self.command.copy()


def _controller(command, u_max):
    controller = SALTRO.__new__(SALTRO)  # skip planner setup
    controller.est_sat = SimpleNamespace(actuators=[SimpleNamespace(u_max=u) for u in u_max])
    controller.active_trajectory = _StubTrajectory(command)
    return controller


def test_find_u_clips_with_the_planning_satellite_when_est_sat_is_none():
    controller = _controller([5.0, -5.0, 0.1], u_max=[1.0, 2.0, 3.0])
    u = controller.find_u(x_hat=None, sens=None, est_sat=None, os_hat=SimpleNamespace(J2000=0.5))
    np.testing.assert_allclose(u, [1.0, -2.0, 0.1])


def test_find_u_clips_with_the_supplied_satellite():
    controller = _controller([5.0, -5.0, 0.1], u_max=[1.0, 2.0, 3.0])
    supplied = SimpleNamespace(actuators=[SimpleNamespace(u_max=u) for u in (0.5, 0.5, 0.05)])
    u = controller.find_u(x_hat=None, sens=None, est_sat=supplied, os_hat=SimpleNamespace(J2000=0.5))
    np.testing.assert_allclose(u, [0.5, -0.5, 0.05])
