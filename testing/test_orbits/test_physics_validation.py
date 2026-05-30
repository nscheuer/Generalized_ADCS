import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants

MU = EarthConstants.mu_e
RE = EarthConstants.R_e
J2 = EarthConstants.J2coeff
EPHEM = Ephemeris()


def make_state(R, V):
    return Orbital_State(ephem=EPHEM, J2000=0.0, R=np.asarray(R, dtype=float), V=np.asarray(V, dtype=float))


def orbital_elements(R, V):
    radius = np.linalg.norm(R)
    speed_sq = V @ V
    momentum = np.cross(R, V)
    momentum_norm = np.linalg.norm(momentum)
    node_vec = np.cross([0.0, 0.0, 1.0], momentum)
    node_norm = np.linalg.norm(node_vec)
    eccentricity_vec = ((speed_sq - MU / radius) * R - (R @ V) * V) / MU
    eccentricity = np.linalg.norm(eccentricity_vec)
    energy = speed_sq / 2.0 - MU / radius
    semi_major_axis = -MU / (2.0 * energy)
    inclination = np.arccos(momentum[2] / momentum_norm)
    raan = np.arctan2(momentum[0], -momentum[1])
    if node_norm > 1e-12 and eccentricity > 1e-12:
        argp = np.arccos(np.clip((node_vec @ eccentricity_vec) / (node_norm * eccentricity), -1.0, 1.0))
        if eccentricity_vec[2] < 0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0
    return {
        "a": semi_major_axis,
        "e": eccentricity,
        "i": inclination,
        "raan": raan,
        "argp": argp,
        "energy": energy,
        "h": momentum,
        "e_vec": eccentricity_vec,
    }


def propagate(state, dt: float, steps: int, *, j2: bool = False):
    current = state
    for _ in range(steps):
        current = state.__class__.propagate_orbit_rk4(current, dt, J2_perturbation_on=j2)
    return current


@pytest.mark.parametrize(
    "R,V,name",
    [
        (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0]), "circular"),
        (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 8.6, 2.0]), "eccentric_inclined"),
    ],
)
def test_two_body_specific_energy_is_conserved(R, V, name):
    state0 = make_state(R, V)
    elements0 = orbital_elements(R, V)
    period = 2.0 * np.pi * np.sqrt(elements0["a"] ** 3 / MU)
    dt = period / 500
    current = state0
    max_drift = 0.0
    for _ in range(3 * 500):
        current = state0.__class__.propagate_orbit_rk4(current, dt, J2_perturbation_on=False)
        elements = orbital_elements(current.R, current.V)
        max_drift = max(max_drift, abs(elements["energy"] - elements0["energy"]) / abs(elements0["energy"]))
    assert max_drift < 1e-6, f"{name}: specific-energy drift {max_drift:.2e}"


@pytest.mark.parametrize(
    "R,V,name",
    [
        (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0]), "circular"),
        (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 8.6, 2.0]), "eccentric_inclined"),
    ],
)
def test_two_body_angular_momentum_is_conserved(R, V, name):
    state0 = make_state(R, V)
    elements0 = orbital_elements(R, V)
    period = 2.0 * np.pi * np.sqrt(elements0["a"] ** 3 / MU)
    dt = period / 500
    current = state0
    max_drift = 0.0
    for _ in range(3 * 500):
        current = state0.__class__.propagate_orbit_rk4(current, dt, J2_perturbation_on=False)
        elements = orbital_elements(current.R, current.V)
        max_drift = max(max_drift, np.linalg.norm(elements["h"] - elements0["h"]) / np.linalg.norm(elements0["h"]))
    assert max_drift < 1e-7, f"{name}: |h| drift {max_drift:.2e}"


@pytest.mark.parametrize(
    "R,V,name",
    [
        (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0]), "circular"),
        (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 8.6, 2.0]), "eccentric_inclined"),
    ],
)
def test_two_body_eccentricity_vector_is_conserved(R, V, name):
    state0 = make_state(R, V)
    elements0 = orbital_elements(R, V)
    period = 2.0 * np.pi * np.sqrt(elements0["a"] ** 3 / MU)
    dt = period / 500
    current = state0
    max_drift = 0.0
    for _ in range(3 * 500):
        current = state0.__class__.propagate_orbit_rk4(current, dt, J2_perturbation_on=False)
        elements = orbital_elements(current.R, current.V)
        max_drift = max(max_drift, np.linalg.norm(elements["e_vec"] - elements0["e_vec"]))
    assert max_drift < 1e-4, f"{name}: eccentricity-vector drift {max_drift:.2e}"


@pytest.mark.parametrize(
    "R,V",
    [
        (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0])),
        (np.array([8000.0, 0.0, 0.0]), np.array([0.0, 7.2, 0.0])),
    ],
)
def test_orbit_closes_in_position_after_kepler_period(R, V):
    state0 = make_state(R, V)
    semi_major_axis = orbital_elements(R, V)["a"]
    period = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / MU)
    propagated = propagate(state0, period / 1200, 1200)
    assert np.linalg.norm(propagated.R - R) < 1.0


@pytest.mark.parametrize(
    "R,V",
    [
        (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0])),
        (np.array([8000.0, 0.0, 0.0]), np.array([0.0, 7.2, 0.0])),
    ],
)
def test_orbit_closes_in_velocity_after_kepler_period(R, V):
    state0 = make_state(R, V)
    semi_major_axis = orbital_elements(R, V)["a"]
    period = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / MU)
    propagated = propagate(state0, period / 1200, 1200)
    assert np.linalg.norm(propagated.V - V) < 1e-3


def test_j2_nodal_regression_matches_first_order_analytic_rate():
    semi_major_axis = RE + 1500.0
    eccentricity = 0.08
    inclination = np.radians(51.6)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    perigee_radius = semi_major_axis * (1.0 - eccentricity)
    perigee_speed = np.sqrt(MU * (2.0 / perigee_radius - 1.0 / semi_major_axis))
    state0 = make_state(
        np.array([perigee_radius, 0.0, 0.0]),
        np.array([0.0, perigee_speed * np.cos(inclination), perigee_speed * np.sin(inclination)]),
    )
    elements0 = orbital_elements(state0.R, state0.V)
    mean_motion = np.sqrt(MU / semi_major_axis**3)
    period = 2.0 * np.pi / mean_motion
    propagated = propagate(state0, 8 * period / (300 * 8), 300 * 8, j2=True)
    elements = orbital_elements(propagated.R, propagated.V)

    def unwrap_rate(end, start, total_time):
        delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
        return delta / total_time

    total_time = 8 * period
    numeric_rate = unwrap_rate(elements["raan"], elements0["raan"], total_time)
    analytic_rate = -1.5 * mean_motion * J2 * (RE / semi_latus_rectum) ** 2 * np.cos(inclination)
    assert analytic_rate < 0.0
    assert abs(numeric_rate - analytic_rate) / abs(analytic_rate) < 0.05


def test_j2_apsidal_precession_matches_first_order_analytic_rate():
    semi_major_axis = RE + 1500.0
    eccentricity = 0.08
    inclination = np.radians(51.6)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    perigee_radius = semi_major_axis * (1.0 - eccentricity)
    perigee_speed = np.sqrt(MU * (2.0 / perigee_radius - 1.0 / semi_major_axis))
    state0 = make_state(
        np.array([perigee_radius, 0.0, 0.0]),
        np.array([0.0, perigee_speed * np.cos(inclination), perigee_speed * np.sin(inclination)]),
    )
    elements0 = orbital_elements(state0.R, state0.V)
    mean_motion = np.sqrt(MU / semi_major_axis**3)
    period = 2.0 * np.pi / mean_motion
    propagated = propagate(state0, 8 * period / (300 * 8), 300 * 8, j2=True)
    elements = orbital_elements(propagated.R, propagated.V)

    def unwrap_rate(end, start, total_time):
        delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
        return delta / total_time

    total_time = 8 * period
    numeric_rate = unwrap_rate(elements["argp"], elements0["argp"], total_time)
    analytic_rate = 0.75 * mean_motion * J2 * (RE / semi_latus_rectum) ** 2 * (5.0 * np.cos(inclination) ** 2 - 1.0)
    assert abs(numeric_rate - analytic_rate) / abs(analytic_rate) < 0.08
