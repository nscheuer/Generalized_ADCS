"""
Unit tests for :class:`~ADCS.satellite_hardware.actuators.Magic_Actuator`.

The "magic" actuator is the simplest possible actuator type: it applies a
direct body-frame torque ``tau = a * u`` with no environmental dependence,
no state coupling, and no storage state. Tests pin:

* ``torque()`` returns ``axis * (u + bias) + noise`` exactly.
* ``dtorq__du`` returns ``axis`` as a ``(1, 3)`` row.
* ``storage_torque`` returns an empty vector.
* All base-state and storage-state derivatives are zero (inherited from
  the base ``Actuator`` zero defaults).
* The ``jacobians()``/``hessians()`` bundled API matches MTQ/RW.

Companion integration test in
``testing/test_controllers/test_planner_magic_actuator.py`` exercises
the actuator through ``Plan_and_Track_LQR``.
"""
from __future__ import annotations

import numpy as np
import pytest

from ADCS.satellite_hardware.actuators import Magic_Actuator
from ADCS.satellite_hardware.errors import Bias
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State


@pytest.fixture(scope="module")
def os0():
    ephem = Ephemeris()
    return Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)


@pytest.fixture
def x_identity():
    """Identity attitude, zero rate (state is unused by Magic torque but
    must be a valid 7-vector for the API)."""
    return np.concatenate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


# ----------------------------------------------------------------------
# torque() basic behaviour
# ----------------------------------------------------------------------

@pytest.mark.parametrize("axis,u,expected", [
    (np.array([1.0, 0.0, 0.0]), 0.5,   np.array([0.5, 0.0, 0.0])),
    (np.array([0.0, 1.0, 0.0]), -0.3,  np.array([0.0, -0.3, 0.0])),
    (np.array([0.0, 0.0, 1.0]), 2.0,   np.array([0.0, 0.0, 2.0])),
    # Off-axis (unit vector at 45 degrees in xy plane)
    (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), 1.0,
     np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)),
])
def test_torque_is_axis_times_command(os0, x_identity, axis, u, expected):
    """``tau = axis * u`` exactly, independent of state and environment."""
    act = Magic_Actuator(axis=axis, max_torque=10.0)
    tau = act.torque(u=u, x=x_identity, os=os0)
    np.testing.assert_allclose(tau, expected, atol=1e-12)


def test_torque_independent_of_attitude(os0):
    """Magic torque doesn't depend on q -- a 180-degree rotation in
    state should produce the same torque as identity.
    """
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    x_identity = np.concatenate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    x_pi_about_y = np.concatenate([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    tau_id = act.torque(u=0.7, x=x_identity, os=os0)
    tau_pi = act.torque(u=0.7, x=x_pi_about_y, os=os0)

    np.testing.assert_allclose(tau_id, tau_pi, atol=1e-14,
                                err_msg="Magic torque must not depend on attitude")


def test_torque_independent_of_rate(os0):
    """Magic torque doesn't depend on omega either."""
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    x_rest = np.concatenate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    x_spin = np.concatenate([[0.5, -0.3, 0.1], [1.0, 0.0, 0.0, 0.0]])
    tau_rest = act.torque(u=1.0, x=x_rest, os=os0)
    tau_spin = act.torque(u=1.0, x=x_spin, os=os0)
    np.testing.assert_allclose(tau_rest, tau_spin, atol=1e-14)


# ----------------------------------------------------------------------
# Bias
# ----------------------------------------------------------------------

def test_torque_with_bias_adds_to_command(os0, x_identity):
    """Bias enters as ``(u + b)``: an additive offset on the command."""
    bias = Bias(bias=0.2, std_bias=0.0)
    act = Magic_Actuator(axis=np.array([1.0, 0.0, 0.0]),
                         max_torque=10.0, bias=bias)
    tau = act.torque(u=0.5, x=x_identity, os=os0)
    # Expected: axis * (u + bias) = [1, 0, 0] * 0.7
    np.testing.assert_allclose(tau, [0.7, 0.0, 0.0], atol=1e-12)


# ----------------------------------------------------------------------
# Saturation warning
# ----------------------------------------------------------------------

def test_command_above_u_max_warns(os0, x_identity):
    act = Magic_Actuator(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0)
    with pytest.warns(UserWarning,
                      match="requested torque exceeds actuation limit"):
        act.torque(u=1.5, x=x_identity, os=os0)


# ----------------------------------------------------------------------
# storage_torque
# ----------------------------------------------------------------------

def test_storage_torque_is_empty(os0, x_identity):
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    st = act.storage_torque(u=1.0, x=x_identity, os=os0)
    assert st.shape == (0,), f"Expected empty storage, got {st.shape}"


# ----------------------------------------------------------------------
# First-order derivatives
# ----------------------------------------------------------------------

@pytest.mark.parametrize("axis", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 2.0, -3.0]) / np.linalg.norm([1.0, 2.0, -3.0]),
])
def test_dtorq_du_equals_axis(os0, x_identity, axis):
    """``dtorq/du = axis`` as a ``(1, 3)`` row, independent of u and state."""
    act = Magic_Actuator(axis=axis, max_torque=10.0)
    dtdu = act.dtorq__du(u=0.5, x=x_identity, os=os0)
    assert dtdu.shape == (1, 3), f"Wrong shape: {dtdu.shape}"
    np.testing.assert_allclose(dtdu.ravel(), axis, atol=1e-12)


def test_dtorq_dbasestate_is_zero(os0, x_identity):
    """Magic torque has no state dependence -> zero Jacobian wrt
    (omega, q)."""
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    dtdx = act.dtorq__dbasestate(u=0.5, x=x_identity, os=os0)
    assert dtdx.shape == (7, 3)
    assert np.allclose(dtdx, 0.0, atol=1e-12)


def test_dtorq_dh_is_empty(os0, x_identity):
    """Magic torque has no momentum-storage state ``h`` -- the
    derivative is the empty ``(0, 3)`` matrix per the base Actuator
    convention."""
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    dtdh = act.dtorq__dh(u=0.5, x=x_identity, os=os0)
    assert dtdh.shape == (0, 3)


# ----------------------------------------------------------------------
# Second-order derivatives (all zero)
# ----------------------------------------------------------------------

def test_all_second_derivatives_are_zero(os0, x_identity):
    """The magic torque is affine in ``u`` and independent of ``x`` and
    ``h``, so all second derivatives vanish.
    """
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    u, x = 0.5, x_identity

    # All second derivatives should be zero (inherited from base
    # Actuator).
    assert np.allclose(act.ddtorq__dudu(u, x, os0), 0.0)
    assert np.allclose(act.ddtorq__dudbasestate(u, x, os0), 0.0)
    assert np.allclose(act.ddtorq__dbasestatedbasestate(u, x, os0), 0.0)


# ----------------------------------------------------------------------
# Bundled jacobians()/hessians() API
# ----------------------------------------------------------------------

def test_jacobians_bundle_matches_individual_methods(os0, x_identity):
    """``jacobians()`` returns ``(dT_du, dT_dx)`` matching the
    individual-method outputs (same API as MTQ/RW)."""
    act = Magic_Actuator(axis=np.array([1.0, 0.0, 0.0]), max_torque=10.0)
    dtdu, dtdx = act.jacobians(u=0.5, x=x_identity, os=os0)
    np.testing.assert_allclose(dtdu, act.dtorq__du(0.5, x_identity, os0))
    np.testing.assert_allclose(dtdx, act.dtorq__dbasestate(0.5, x_identity, os0))


def test_hessians_bundle_returns_zeros_of_correct_shape(os0, x_identity):
    act = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=10.0)
    ddt_du_dx, ddt_dx2 = act.hessians(u=0.5, x=x_identity, os=os0)
    assert ddt_du_dx.shape == (1, 7, 3)
    assert ddt_dx2.shape == (7, 7, 3)
    assert np.allclose(ddt_du_dx, 0.0)
    assert np.allclose(ddt_dx2, 0.0)


# ----------------------------------------------------------------------
# Numerical FD check on dtorq/du
# ----------------------------------------------------------------------

def test_dtorq_du_matches_finite_difference(os0, x_identity):
    """Sanity: central FD of ``torque(u)`` equals ``axis`` to machine
    precision -- since the torque is exactly affine in ``u`` the FD
    has no truncation error."""
    axis = np.array([0.3, -0.4, 0.7]) / np.linalg.norm([0.3, -0.4, 0.7])
    act = Magic_Actuator(axis=axis, max_torque=10.0)
    eps = 1e-6
    tau_p = act.torque(u=eps, x=x_identity, os=os0)
    tau_m = act.torque(u=-eps, x=x_identity, os=os0)
    fd = (tau_p - tau_m) / (2.0 * eps)
    np.testing.assert_allclose(fd, axis, atol=1e-10)
