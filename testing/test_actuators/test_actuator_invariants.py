"""
Physical-invariant regression tests for ADCS actuator models.

Covers four verified bugs in ADCS/satellite_hardware/:

1. RW violated Newton's 3rd law under noise/bias: torque() and
   storage_torque() independently resampled noise/bias, so the body torque
   and the wheel-reaction torque used different random draws and angular
   momentum was created every step. They must now share one coherent draw
   per (step/state), call-order independent.
2. No saturation clamping: MTQ.torque and RW.torque/storage_torque only
   warned on over-limit commands but still scaled with the raw command.
   They must now clamp to +/- u_max (sign preserved, below-limit unchanged).
3. RW momentum-saturation check was sign-blind and array-broken
   (``if h > self.h_max``): negative saturation missed, vector h raised.
   It must compare by magnitude, clamp at +/- h_max, and accept vector h.
4. Base Actuator.torque returned ``np.ndarray([0, 0, 0])`` (uninitialized
   shape-(0,0,0) garbage) instead of the zero vector ``np.zeros(3)``.

All tests seed the RNG for determinism.
"""

import sys
import os
import inspect
import numpy as np
import pytest

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise, ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec


WORKTREE = os.path.abspath(os.path.join(__file__, "../../.."))


def _make_os(j2000=0.22):
    ephem = Ephemeris()
    B_ECI = 1e-5 * np.array([1.0, 2.0, -3.0]) / np.linalg.norm([1.0, 2.0, -3.0])
    return Orbital_State(
        ephem=ephem,
        J2000=j2000,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=B_ECI,
    )


def test_worktree_shadowing():
    """The ADCS package under test must be the one in THIS worktree."""
    loaded = os.path.abspath(inspect.getfile(ADCS))
    assert loaded.startswith(WORKTREE), (
        f"ADCS loaded from {loaded}, not from worktree {WORKTREE}; "
        f"PYTHONPATH is not shadowing the editable install."
    )


# ---------------------------------------------------------------------------
# Bug 4: base Actuator.torque must return the zero vector, not garbage.
# ---------------------------------------------------------------------------
def test_base_actuator_torque_is_zero_vector():
    act = Actuator(axis=np.array([1.0, 0.0, 0.0]), u_max=1.0)
    os_ = _make_os()
    tau = act.torque(u=0.5, x=np.array([0, 0, 0, 1, 0, 0, 0], dtype=float), os=os_)
    tau = np.asarray(tau)
    assert tau.shape == (3,)
    assert tau.dtype.kind == "f"
    assert np.array_equal(tau, np.zeros(3))


# ---------------------------------------------------------------------------
# Bug 2: MTQ saturation clamping.
# ---------------------------------------------------------------------------
def test_mtq_saturation_clamps_command_both_signs():
    np.random.seed(12345)
    ax = random_n_unit_vec(3)
    u_max = 2.0
    mtq = MTQ(axis=ax, max_torque=u_max)  # no bias / noise
    os_ = _make_os()
    x0 = np.hstack((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))

    # Reference torque is linear in the (clamped) command for the MTQ.
    tau_at = lambda u: mtq.torque(u=u, x=x0, os=os_)

    # Below limit: unchanged (linear, exact).
    u_lo = 0.5
    assert np.allclose(tau_at(u_lo), tau_at(0.0) + u_lo / u_max * tau_at(u_max))

    tau_pos_lim = tau_at(u_max)
    tau_neg_lim = tau_at(-u_max)

    # Beyond +u_max -> identical to the +u_max command (clamped).
    with pytest.warns(UserWarning):
        tau_over_pos = tau_at(10.0 * u_max)
    assert np.allclose(tau_over_pos, tau_pos_lim)
    assert np.isclose(np.linalg.norm(tau_over_pos), np.linalg.norm(tau_pos_lim))

    # Beyond -u_max -> identical to the -u_max command (clamped), opposite sign.
    with pytest.warns(UserWarning):
        tau_over_neg = tau_at(-10.0 * u_max)
    assert np.allclose(tau_over_neg, tau_neg_lim)
    assert np.allclose(tau_over_neg, -tau_pos_lim)


# ---------------------------------------------------------------------------
# Bug 2: RW saturation clamping (torque and storage_torque).
# ---------------------------------------------------------------------------
def test_rw_saturation_clamps_command_both_signs():
    np.random.seed(2024)
    ax = random_n_unit_vec(3)
    u_max = 0.05
    rw = RW(axis=ax, max_torque=u_max, J=0.01, h=0.0, h_max=0.2)  # no bias/noise
    os_ = _make_os()
    x0 = np.hstack((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    axis_unit = rw.axis

    # Below limit unchanged.
    u_lo = 0.5 * u_max
    tau_lo = rw.torque(u=u_lo, x=x0, os=os_)
    assert np.allclose(tau_lo, axis_unit * u_lo)
    assert np.isclose(rw.storage_torque(u=u_lo, x=x0, os=os_), -u_lo)

    # Beyond +u_max -> magnitude == limit, sign along +axis.
    with pytest.warns(UserWarning):
        tau_pos = rw.torque(u=7.3 * u_max, x=x0, os=os_)
    assert np.isclose(np.linalg.norm(tau_pos), u_max)
    assert np.allclose(tau_pos, axis_unit * u_max)
    with pytest.warns(UserWarning):
        st_pos = rw.storage_torque(u=7.3 * u_max, x=x0, os=os_)
    assert np.isclose(st_pos, -u_max)

    # Beyond -u_max -> magnitude == limit, sign along -axis.
    with pytest.warns(UserWarning):
        tau_neg = rw.torque(u=-9.1 * u_max, x=x0, os=os_)
    assert np.isclose(np.linalg.norm(tau_neg), u_max)
    assert np.allclose(tau_neg, -axis_unit * u_max)
    with pytest.warns(UserWarning):
        st_neg = rw.storage_torque(u=-9.1 * u_max, x=x0, os=os_)
    assert np.isclose(st_neg, u_max)


# ---------------------------------------------------------------------------
# Bug 3: RW momentum saturation must be sign-aware and array-safe.
# ---------------------------------------------------------------------------
def test_rw_update_momentum_clamps_both_signs_scalar():
    rw = RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0, J=0.01, h=0.0, h_max=0.1)

    # Positive over-saturation: clamped to +h_max, warned.
    with pytest.warns(UserWarning):
        rw.update_momentum(0.5)
    assert np.isclose(rw.h, 0.1)

    # Negative over-saturation: previously sign-blind ``h > h_max`` MISSED this.
    with pytest.warns(UserWarning):
        rw.update_momentum(-0.5)
    assert np.isclose(rw.h, -0.1)

    # Within bound: unchanged, no clamp.
    rw.update_momentum(0.07)
    assert np.isclose(rw.h, 0.07)
    rw.update_momentum(-0.07)
    assert np.isclose(rw.h, -0.07)


def test_rw_update_momentum_accepts_vector_without_raising():
    rw = RW(
        axis=np.array([0.0, 0.0, 1.0]),
        max_torque=1.0,
        J=0.01,
        h=np.zeros(3),
        h_max=np.array([0.1, 0.1, 0.1]),
    )
    # Previously ``if h > self.h_max`` raised an ambiguous-truth ValueError.
    big = np.array([0.3, -0.4, 0.0])  # magnitude 0.5 > h_max magnitude 0.1
    with pytest.warns(UserWarning):
        rw.update_momentum(big)
    h = np.asarray(rw.h)
    assert h.shape == (3,)
    # Clamped to magnitude h_max, direction preserved.
    assert np.isclose(np.linalg.norm(h), 0.1)
    assert np.allclose(h / np.linalg.norm(h), big / np.linalg.norm(big))

    # Small vector: untouched.
    small = np.array([0.01, 0.02, 0.0])
    rw.update_momentum(small)
    assert np.allclose(np.asarray(rw.h), small)


# ---------------------------------------------------------------------------
# Bug 1: Newton's third law / angular-momentum conservation under noise+bias.
# ---------------------------------------------------------------------------
def test_rw_newton_third_law_under_noise_and_bias():
    """
    Over many integration steps, the body torque projected on the wheel axis
    plus the wheel-reaction (storage) torque must cancel to ~machine zero,
    independent of the order in which torque()/storage_torque() are called.
    """
    np.random.seed(98765)
    ax = random_n_unit_vec(3)
    rw = RW(
        axis=ax,
        max_torque=10.0,
        J=0.02,
        h=0.0,
        h_max=5.0,
        bias=Bias(bias=0.3, std_bias=1.0),
        noise=Noise(noise=0.0, std_noise=1.0),
    )
    axis_unit = rw.axis
    os_ = _make_os()
    x0 = np.hstack((0.01 * np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])))
    u = 0.4

    n_steps = 200
    max_residual = 0.0
    for _ in range(n_steps):
        os_.J2000 += TimeConstants.sec2cent  # advance one step

        # Call order A: torque() then storage_torque()
        tau_body = rw.torque(u=u, x=x0, os=os_)
        st = rw.storage_torque(u=u, x=x0, os=os_)
        residual = float(np.dot(tau_body, axis_unit) + st)
        max_residual = max(max_residual, abs(residual))

    assert max_residual < 1e-12, (
        f"Newton's 3rd law violated: max |tau_body.axis + storage_torque| "
        f"= {max_residual} (expected ~0)."
    )


def test_rw_reaction_is_call_order_independent():
    """
    Within a single (step, command, dmode), torque() and storage_torque() must
    use ONE shared stochastic draw, so Newton's 3rd law holds regardless of
    which is called first, and a repeated call within the step does not
    resample. Tested on a single RW (Noise/Bias share the global np.random
    stream, so two interleaved RWs would diverge by RNG, not by call order --
    that is not a valid order-independence test).
    """
    np.random.seed(13)
    ax = random_n_unit_vec(3)
    rw = RW(
        axis=ax,
        max_torque=10.0,
        J=0.02,
        h=0.0,
        h_max=5.0,
        bias=Bias(bias=0.3, std_bias=1.0),
        noise=Noise(noise=0.0, std_noise=1.0),
    )
    axis_unit = rw.axis
    x0 = np.hstack((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    u = 0.4
    os_ = _make_os()

    for k in range(50):
        os_.J2000 += TimeConstants.sec2cent
        if k % 2 == 0:
            # torque() first, then storage_torque(), then torque() again.
            tau1 = rw.torque(u=u, x=x0, os=os_)
            st = rw.storage_torque(u=u, x=x0, os=os_)
            tau2 = rw.torque(u=u, x=x0, os=os_)
        else:
            # storage_torque() first -> the reverse order.
            st = rw.storage_torque(u=u, x=x0, os=os_)
            tau1 = rw.torque(u=u, x=x0, os=os_)
            tau2 = rw.torque(u=u, x=x0, os=os_)
        # Repeated call within the step is cached (no resample).
        assert np.array_equal(tau1, tau2)
        # Reaction exactly opposite the applied body torque, either order.
        assert abs(float(np.dot(tau1, axis_unit)) + st) < 1e-12


def test_rw_dynamics_step_conserves_total_angular_momentum():
    """
    Within a single dynamics step the spacecraft+wheel system must conserve
    angular momentum under noise/bias: the impulse on the body equals minus
    the impulse on the wheel along the spin axis. This is exactly the
    invariant that the independent-resampling bug broke (verified to leak
    ~0.1-3 N.m per step at std_noise=1).
    """
    np.random.seed(555)
    ax = random_n_unit_vec(3)
    rw = RW(
        axis=ax,
        max_torque=10.0,
        J=0.02,
        h=0.0,
        h_max=5.0,
        bias=Bias(bias=0.3, std_bias=1.0),
        noise=Noise(noise=0.0, std_noise=1.0),
    )
    axis_unit = rw.axis
    os_ = _make_os()
    x0 = np.hstack((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

    leaks = []
    for _ in range(50):
        os_.J2000 += TimeConstants.sec2cent
        tau_body = rw.torque(u=0.4, x=x0, os=os_, dmode=dmode)
        st = rw.storage_torque(u=0.4, x=x0, os=os_, dmode=dmode)
        leaks.append(float(np.dot(tau_body, axis_unit) + st))

    leaks = np.asarray(leaks)
    assert np.max(np.abs(leaks)) < 1e-12, (
        f"angular momentum leaked: max |leak| = {np.max(np.abs(leaks))}"
    )
