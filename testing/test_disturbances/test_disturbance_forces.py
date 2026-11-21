"""The force side of the disturbance interface.

Every disturbance exposes ``force``, ``force_qjac`` and ``force_qqhess`` the
same way it exposes the torque trio.  Torque-only disturbances inherit zeros
from the base class, and ``Satellite.dist_forces`` sums whatever the
registered disturbances return.
"""

import numpy as np

from ADCS.satellite_hardware.disturbances import (
    Dipole_Disturbance,
    Disturbance,
    GG_Disturbance,
    Prop_Disturbance,
    Torque_Disturbance,
)
from testing.test_disturbances._helpers import (
    make_orbital_state,
    make_satellite,
    make_state,
)


class _ConstantForce(Disturbance):
    """A disturbance that pushes with a fixed body-frame force and needs ``sat``."""

    def __init__(self, force):
        super().__init__()
        self._force = np.asarray(force, dtype=float)

    def torque(self, sat, x, os):
        return np.zeros(3)

    def force(self, sat, x, os):
        assert sat is not None
        return self._force


class _ConstantForceNoSat(Disturbance):
    """Same, but with the two-argument ``(x, os)`` signature."""

    def __init__(self, force):
        super().__init__()
        self._force = np.asarray(force, dtype=float)

    def torque(self, x, os):
        return np.zeros(3)

    def force(self, x, os):
        return self._force


def test_base_class_force_and_derivatives_are_zero():
    base = Disturbance()
    x = make_state()
    os = make_orbital_state()

    assert np.array_equal(base.force(x=x, os=os), np.zeros(3))
    assert np.array_equal(base.force_qjac(x=x, os=os), np.zeros((4, 3)))
    assert np.array_equal(base.force_qqhess(x=x, os=os), np.zeros((4, 4, 3)))


def test_torque_only_disturbances_inherit_zero_force():
    sat = make_satellite()
    x = make_state()
    os = make_orbital_state()
    torque_only = [
        Prop_Disturbance(np.array([1e-6, -2e-6, 3e-6])),
        Dipole_Disturbance(np.array([0.01, 0.02, -0.03])),
        GG_Disturbance(),
        Torque_Disturbance(np.array([1e-7, 0.0, 0.0])),
    ]
    for dist in torque_only:
        assert np.array_equal(dist.force(sat=sat, x=x, os=os), np.zeros(3)), type(dist)
        assert dist.force_qjac(sat=sat, x=x, os=os).shape == (4, 3)
        assert dist.force_qqhess(sat=sat, x=x, os=os).shape == (4, 4, 3)


def test_dist_forces_sums_both_call_signatures():
    f1 = np.array([1.0e-5, 0.0, -2.0e-5])
    f2 = np.array([0.0, 3.0e-5, 1.0e-5])
    sat = make_satellite(
        disturbances=[
            _ConstantForce(f1),
            _ConstantForceNoSat(f2),
            Prop_Disturbance(np.array([1e-6, 0.0, 0.0])),
        ]
    )
    x = make_state()
    os = make_orbital_state()

    assert np.allclose(sat.dist_forces(x=x, os=os), f1 + f2)


def test_dist_forces_is_zero_without_force_producing_disturbances():
    sat = make_satellite(disturbances=[Prop_Disturbance(np.array([1e-6, 0.0, 0.0])), GG_Disturbance()])
    x = make_state()
    os = make_orbital_state()

    assert np.array_equal(sat.dist_forces(x=x, os=os), np.zeros(3))


def test_dist_forces_does_not_change_attitude_dynamics():
    sat_with = make_satellite(disturbances=[_ConstantForce(np.array([0.1, 0.2, 0.3]))])
    sat_without = make_satellite(disturbances=[])
    x = make_state(w=np.array([0.01, -0.02, 0.03]))
    os = make_orbital_state()

    xdot_with = sat_with.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)
    xdot_without = sat_without.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)

    assert np.allclose(xdot_with, xdot_without)
