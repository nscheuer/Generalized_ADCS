"""Regression checks for osculating elements computed by Orbital_State.

Reference Cartesian states are generated from eccentric anomaly in the orbital
plane, independently of the state-to-elements conversion under test.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from skyfield.api import load

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants


pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
MU = EarthConstants.mu_e
J2000 = 0.22


@pytest.fixture(scope="module")
def make_state():
    # Exercise the real constructor without downloading a planetary ephemeris:
    # only its timescale is needed when Sun, magnetic field and density are given.
    ephem = SimpleNamespace(ts=load.timescale(builtin=True))

    def make(R, V):
        return Orbital_State(
            ephem=ephem,
            J2000=J2000,
            R=np.asarray(R, dtype=float),
            V=np.asarray(V, dtype=float),
            S=np.array([1.5e8, 0.0, 0.0]),
            B=np.array([2e-5, -1e-5, 3e-5]),
            rho=0.0,
        )

    return make


def cartesian_from_elements(a, e, inclination, node, periapsis, E):
    """Convert a known ellipse to Cartesian state; all angles are radians."""
    def rz(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    c, s = np.cos(inclination), np.sin(inclination)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    frame = rz(node) @ rx @ rz(periapsis)
    b = a * np.sqrt(1.0 - e**2)
    position = np.array([a * (np.cos(E) - e), b * np.sin(E), 0.0])
    dE_dt = np.sqrt(MU / a**3) / (1.0 - e * np.cos(E))
    velocity = dE_dt * np.array([-a * np.sin(E), b * np.cos(E), 0.0])
    return frame @ position, frame @ velocity, frame


def assert_angle(actual, expected):
    """Compare angles modulo a full revolution, including the 0/2pi boundary."""
    difference = np.arctan2(np.sin(actual - expected), np.cos(actual - expected))
    assert difference == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize("e", [0.05, 0.6])
@pytest.mark.parametrize(
    "orientation_deg", [(40.0, 30.0, 240.0), (120.0, 300.0, 40.0)],
    ids=["prograde-periapsis-quadrant-III", "retrograde-node-quadrant-IV"],
)
@pytest.mark.parametrize("E_deg", [0.0, 60.0, 180.0, 300.0])
def test_known_elliptic_elements(make_state, e, orientation_deg, E_deg):
    a = 10000.0
    inclination, node, periapsis = np.deg2rad(orientation_deg)
    E = np.deg2rad(E_deg)
    R, V, frame = cartesian_from_elements(a, e, inclination, node, periapsis, E)
    state = make_state(R, V)

    # Expected anomalies come from the reference ellipse, not the recovered e/R.
    nu = np.arctan2(np.sqrt(1.0 - e**2) * np.sin(E), np.cos(E) - e) % (2 * np.pi)
    M = E - e * np.sin(E)
    mean_motion = np.sqrt(MU / a**3)
    expected = {
        "semi_major_axis": a,
        "semi_minor_axis": a * np.sqrt(1.0 - e**2),
        "semi_parameter": a * (1.0 - e**2),
        "eccentricity": e,
        "periapsis": a * (1.0 - e),
        "apoapsis": a * (1.0 + e),
        "mean_motion": mean_motion,
        "orbital_period": 2 * np.pi / mean_motion,
    }
    for name, value in expected.items():
        assert getattr(state, name) == pytest.approx(value, rel=1e-11), name
    np.testing.assert_allclose(state.eccentricity_vector, frame[:, 0] * e, atol=1e-13)

    angles = {
        "inclination": inclination,
        "longitude_of_ascending_node": node,
        "argument_of_periapsis": periapsis,
        "longitude_of_periapsis": node + periapsis,
        "eccentric_anomaly_at_epoch": E,
        "true_anomaly_at_epoch": nu,
        "mean_anomaly_at_epoch": M,
        "mean_longitude_at_epoch": node + periapsis + M,
        "true_longitude_at_epoch": node + periapsis + nu,
        "mean_argument_of_latitude_at_epoch": periapsis + M,
        "argument_of_latitude_at_epoch": periapsis + nu,
    }
    for name, value in angles.items():
        assert_angle(getattr(state, name), value)

    assert state.epoch == pytest.approx(2451545.0 + J2000 * 36525.0, abs=1e-9)
    elapsed_seconds = (state.epoch - state.time_of_periapsis_passage) * 86400.0
    # Julian dates around 2.46e6 resolve time differences to about 40 microseconds.
    if E_deg == 0.0:
        # At periapsis, roundoff can wrap M to either 0 or 2pi; both passage
        # epochs describe the same osculating ellipse, one period apart.
        assert min(abs(elapsed_seconds), abs(elapsed_seconds - state.orbital_period)) < 5e-5
    else:
        assert elapsed_seconds == pytest.approx(M / mean_motion, abs=5e-5)
    assert state.mass_of_the_two_bodies == pytest.approx(5.972168489579432e24, rel=1e-12)


@pytest.mark.parametrize(
    "e,inclination_deg,expected_node_deg,expected_periapsis_deg,circular_nu_deg",
    [
        (0.0, 40.0, 30.0, 0.0, 110.0),
        (0.0, 0.0, 0.0, 0.0, 140.0),
        (0.0, 180.0, 0.0, 0.0, 80.0),
        (0.2, 0.0, 0.0, 70.0, None),
        (0.2, 180.0, 0.0, 10.0, None),
    ],
    ids=["circular-inclined", "circular-equatorial", "circular-retrograde",
         "eccentric-equatorial", "eccentric-retrograde"],
)
def test_circular_and_equatorial_conventions(
    make_state, e, inclination_deg, expected_node_deg, expected_periapsis_deg, circular_nu_deg
):
    inclination, node, periapsis, E = np.deg2rad([inclination_deg, 30.0, 40.0, 70.0])
    R, V, _ = cartesian_from_elements(10000.0, e, inclination, node, periapsis, E)
    state = make_state(R, V)

    assert_angle(state.longitude_of_ascending_node, np.deg2rad(expected_node_deg))
    assert_angle(state.argument_of_periapsis, np.deg2rad(expected_periapsis_deg))
    if circular_nu_deg is not None:
        assert_angle(state.true_anomaly_at_epoch, np.deg2rad(circular_nu_deg))
        assert_angle(state.mean_anomaly_at_epoch, state.true_anomaly_at_epoch)
    else:
        assert_angle(state.eccentric_anomaly_at_epoch, E)

    reconstructed_R, reconstructed_V, _ = cartesian_from_elements(
        state.semi_major_axis, state.eccentricity, state.inclination,
        state.longitude_of_ascending_node, state.argument_of_periapsis,
        state.eccentric_anomaly_at_epoch,
    )
    np.testing.assert_allclose(reconstructed_R, R, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(reconstructed_V, V, rtol=0.0, atol=1e-11)


def test_keplerian_dictionary_exposes_elements_with_correct_shapes(make_state):
    R, V, _ = cartesian_from_elements(10000.0, 0.2, *np.deg2rad([40, 30, 240, 300]))
    state = make_state(R, V)
    aliases = {
        "true_anomaly": "true_anomaly_at_epoch",
        "eccentric_anomaly": "eccentric_anomaly_at_epoch",
    }
    expected_keys = {
        "semi_major_axis", "semi_minor_axis", "semi_parameter", "eccentricity",
        "eccentricity_vector", "inclination", "longitude_of_ascending_node",
        "argument_of_periapsis", "longitude_of_periapsis", "mean_motion",
        "orbital_period", "epoch", "time_of_periapsis_passage",
        "mean_longitude_at_epoch", "true_longitude_at_epoch",
        "mean_argument_of_latitude_at_epoch", "argument_of_latitude_at_epoch",
        "true_anomaly", "eccentric_anomaly", "mean_anomaly_at_epoch",
        "apoapsis", "periapsis", "mass_of_the_two_bodies",
    }
    assert set(state.keplerian_elements) == expected_keys
    for name, value in state.keplerian_elements.items():
        assert np.shape(value) == ((3,) if name == "eccentricity_vector" else ()), name
        np.testing.assert_equal(value, getattr(state, aliases.get(name, name)))


@pytest.mark.parametrize(
    "kind,velocity",
    [
        ("hyperbolic", [0.0, 11.0, 2.0]),
        ("parabolic", [0.0, np.sqrt(2 * MU / 7000.0), 0.0]),
        ("radial", [1.0, 0.0, 0.0]),
        ("stationary", [0.0, 0.0, 0.0]),
    ],
    ids=["hyperbolic", "parabolic", "radial", "stationary"],
)
def test_undefined_elements_for_unbound_and_radial_states(make_state, kind, velocity):
    state = make_state([7000.0, 0.0, 0.0], velocity)
    for name in (
        "apoapsis", "mean_motion", "orbital_period", "eccentric_anomaly_at_epoch",
        "mean_anomaly_at_epoch", "time_of_periapsis_passage", "mean_longitude_at_epoch",
        "mean_argument_of_latitude_at_epoch",
    ):
        assert np.isnan(getattr(state, name)), name
    if kind == "hyperbolic":
        assert state.eccentricity > 1
        assert state.semi_major_axis < 0
        assert state.semi_minor_axis > 0
        assert state.semi_parameter > 0
        assert state.periapsis == pytest.approx(7000.0)
    elif kind == "parabolic":
        assert state.eccentricity == pytest.approx(1.0)
        assert np.isposinf(state.semi_major_axis)
        assert np.isnan(state.semi_minor_axis)
        assert state.semi_parameter == pytest.approx(14000.0)
        assert state.periapsis == pytest.approx(7000.0)
    else:
        assert state.semi_parameter == 0
        for name in ("inclination", "longitude_of_ascending_node",
                     "argument_of_periapsis", "true_anomaly_at_epoch"):
            assert np.isnan(getattr(state, name)), name


@pytest.mark.parametrize("velocity", [[0.0, 0.0, 0.0], [0.0, 7.5, 0.0]])
def test_zero_position_placeholder_preserves_frame_conversion(make_state, velocity):
    reference = make_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    # Geographic/environment values at the origin were already undefined before
    # Keplerian elements were added. Only epoch-based frame conversion is used.
    with np.errstate(invalid="ignore", divide="ignore"):
        placeholder = make_state([0.0, 0.0, 0.0], velocity)

    np.testing.assert_array_equal(placeholder.R, np.zeros(3))
    np.testing.assert_array_equal(placeholder.V, velocity)
    for axis in np.eye(3):
        np.testing.assert_allclose(
            placeholder.ecef_to_eci(axis), reference.ecef_to_eci(axis), atol=1e-12
        )
        np.testing.assert_allclose(
            placeholder.eci_to_ecef(axis), reference.eci_to_ecef(axis), atol=1e-12
        )
    for name, value in placeholder.keplerian_elements.items():
        if name in ("epoch", "mass_of_the_two_bodies"):
            assert np.isfinite(value), name
        else:
            assert np.all(np.isnan(value)), name


@pytest.mark.parametrize(
    "position,velocity",
    [
        ([np.nan, 0.0, 0.0], [0.0, 7.5, 0.0]),
        ([7000.0, 0.0, 0.0], [0.0, np.inf, 0.0]),
    ],
    ids=["nonfinite-position", "nonfinite-velocity"],
)
def test_invalid_states_are_rejected(make_state, position, velocity):
    with pytest.raises(ValueError, match="finite R and V"):
        make_state(position, velocity)
