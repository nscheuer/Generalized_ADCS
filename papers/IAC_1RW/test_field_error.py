"""Tests for the estimator-vs-plant field-model error (campaign §3)."""

import os as _os
import sys

import numpy as np
import pytest

sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State

from papers.IAC_1RW._field_error import FieldErrorModel, wrap_os_for_gnc

_DEG = np.pi / 180.0


def make_os() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([6778.0, 0.0, 0.0]),
        V=np.array([0.0, 7.67, 0.0]),
    )


def test_deterministic_model_hits_the_requested_error_exactly():
    m = FieldErrorModel(direction_deg=4.0, magnitude_frac=0.04,
                        rng=np.random.default_rng(0), deterministic=True)
    B = np.array([2e-5, -1e-5, 3e-5])
    Bp = m.apply(B)

    ang = np.degrees(np.arccos(np.clip(
        np.dot(B, Bp) / (np.linalg.norm(B) * np.linalg.norm(Bp)), -1.0, 1.0)))
    assert np.isclose(ang, 4.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(Bp) / np.linalg.norm(B), 1.04, rtol=1e-12)


def test_random_draws_match_the_requested_sigmas():
    rng = np.random.default_rng(4321)
    B = np.array([2e-5, -1e-5, 3e-5])
    angs, mags = [], []
    for _ in range(3000):
        m = FieldErrorModel(direction_deg=4.0, magnitude_frac=0.04, rng=rng)
        Bp = m.apply(B)
        angs.append(np.degrees(np.arccos(np.clip(
            np.dot(B, Bp) / (np.linalg.norm(B) * np.linalg.norm(Bp)), -1.0, 1.0))))
        mags.append(np.linalg.norm(Bp) / np.linalg.norm(B) - 1.0)
    # |theta| is folded-normal: E|theta| = sigma*sqrt(2/pi) = 0.798*sigma
    assert np.isclose(np.mean(angs), 4.0 * np.sqrt(2 / np.pi), rtol=0.08)
    assert np.isclose(np.std(mags), 0.04, rtol=0.08)


def test_direction_error_is_exactly_theta_for_every_field_direction():
    """The realized error must not depend on where in the orbit B happens to point.

    Rotating about a *fixed* axis would move a B parallel to that axis not at all, so the
    orbit-average error would fall below the quoted figure. The axis is built per call as
    B x reference instead, which pins the realized angle to theta everywhere.
    """
    m = FieldErrorModel(direction_deg=4.0, magnitude_frac=0.0,
                        rng=np.random.default_rng(19), deterministic=True)
    rng = np.random.default_rng(99)
    for _ in range(500):
        B = rng.standard_normal(3) * 3e-5
        Bp = m.apply(B)
        ang = np.degrees(np.arccos(np.clip(
            np.dot(B, Bp) / (np.linalg.norm(B) * np.linalg.norm(Bp)), -1.0, 1.0)))
        assert np.isclose(ang, 4.0, atol=1e-8), ang

    # Including the degenerate case where B is parallel to the frozen reference.
    B = m.reference * 3e-5
    Bp = m.apply(B)
    ang = np.degrees(np.arccos(np.clip(
        np.dot(B, Bp) / (np.linalg.norm(B) * np.linalg.norm(Bp)), -1.0, 1.0)))
    assert np.isclose(ang, 4.0, atol=1e-8), ang


def test_error_is_frozen_within_a_trial():
    """A field-model error is systematic; resampling per step would average it away."""
    m = FieldErrorModel(rng=np.random.default_rng(7))
    B = np.array([2e-5, -1e-5, 3e-5])
    first = m.apply(B)
    for _ in range(100):
        np.testing.assert_allclose(m.apply(B), first, rtol=0, atol=0)


def test_wrapper_perturbs_only_the_field():
    o = make_os()
    m = FieldErrorModel(rng=np.random.default_rng(11), deterministic=True)
    w = wrap_os_for_gnc(o, m)

    assert not np.allclose(w.B, o.B), "field must actually change"
    np.testing.assert_allclose(w.R, o.R)
    np.testing.assert_allclose(w.V, o.V)
    np.testing.assert_allclose(w.S, o.S)
    assert w.J2000 == o.J2000
    np.testing.assert_allclose(w.rho, o.rho)


def test_wrapper_does_not_mutate_the_plant_state():
    o = make_os()
    B_before = o.B.copy()
    wrap_os_for_gnc(o, FieldErrorModel(rng=np.random.default_rng(2)))
    np.testing.assert_allclose(o.B, B_before, rtol=0, atol=0)


def test_body_frame_field_follows_the_perturbation():
    """B is the single source of truth: the derived body-frame field must move with it."""
    o = make_os()
    m = FieldErrorModel(direction_deg=4.0, magnitude_frac=0.04,
                        rng=np.random.default_rng(5), deterministic=True)
    w = wrap_os_for_gnc(o, m)

    x = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])
    b_true = o.get_state_vector(x=x)["b"]
    b_est = w.get_state_vector(x=x)["b"]

    assert not np.allclose(b_est, b_true)
    ang = np.degrees(np.arccos(np.clip(
        np.dot(b_true, b_est) / (np.linalg.norm(b_true) * np.linalg.norm(b_est)), -1.0, 1.0)))
    assert np.isclose(ang, 4.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(b_est) / np.linalg.norm(b_true), 1.04, rtol=1e-9)


def test_zero_error_is_a_no_op():
    o = make_os()
    m = FieldErrorModel(direction_deg=0.0, magnitude_frac=0.0,
                        rng=np.random.default_rng(1), deterministic=True)
    w = wrap_os_for_gnc(o, m)
    np.testing.assert_allclose(w.B, o.B, rtol=1e-12, atol=0)


def test_realized_values_are_reportable():
    m = FieldErrorModel(direction_deg=4.0, magnitude_frac=0.04,
                        rng=np.random.default_rng(3))
    assert m.realized_direction_deg >= 0.0
    assert np.isclose(m.realized_magnitude_frac, m.scale - 1.0)
