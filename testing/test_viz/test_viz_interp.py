"""Quaternion continuity and interpolation helpers for the viewer."""

import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.viz.interp import hermite_position, interp_quat, quat_continuity


def _spin(t, rate=0.3, axis=(0.0, 0.0, 1.0)):
    axis = normalize(np.asarray(axis, dtype=float))
    return np.array([np.concatenate([[np.cos(rate * tk / 2)], np.sin(rate * tk / 2) * axis]) for tk in t])


def test_continuity_removes_sign_flips():
    q = _spin(np.arange(50.0))
    flipped = q * np.where(np.random.default_rng(1).random(50) < 0.5, -1.0, 1.0)[:, None]
    fixed = quat_continuity(flipped)
    assert np.all(np.einsum("ij,ij->i", fixed[1:], fixed[:-1]) > 0)
    np.testing.assert_allclose(np.abs(fixed), np.abs(q))
    assert np.array_equal(fixed[0], flipped[0])


def test_slerp_recovers_constant_rate_rotation():
    t = np.arange(0.0, 20.0, 2.0)
    q = quat_continuity(_spin(t, axis=(1, 2, 3)))
    q[3] *= -1.0  # interpolation must not depend on the sign stored
    t_new = np.linspace(0.0, 18.0, 37)
    out = interp_quat(t, q, t_new)
    ref = _spin(t_new, axis=(1, 2, 3))
    np.testing.assert_allclose(np.abs(np.einsum("ij,ij->i", out, ref)), 1.0, atol=1e-12)


def test_hermite_beats_linear_on_a_leo_orbit():
    mu, a = EarthConstants.mu_e, 7000.0
    n = np.sqrt(mu / a**3)
    t = np.arange(0.0, 600.0, 30.0)

    def rv(tt):
        th = n * tt
        return (a * np.column_stack([np.cos(th), np.sin(th), 0 * th]),
                a * n * np.column_stack([-np.sin(th), np.cos(th), 0 * th]))

    r, v = rv(t)
    t_new = t[:-1] + 15.0
    exact, _ = rv(t_new)
    herm = hermite_position(t, r, v, t_new)
    lin = 0.5 * (r[:-1] + r[1:])
    herm_err = np.linalg.norm(herm - exact, axis=1).max() * 1e3  # m
    lin_err = np.linalg.norm(lin - exact, axis=1).max() * 1e3
    assert lin_err > 500.0      # ~1 km chord error at 30 s
    assert herm_err < 0.05      # ~2 cm


def test_queries_outside_samples_are_rejected():
    with pytest.raises(ValueError):
        interp_quat(np.array([0.0, 1.0]), _spin([0.0, 1.0]), np.array([2.0]))
