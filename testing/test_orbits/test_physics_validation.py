"""Fast, independent regression checks for two-body and J2 orbit physics.

These are deliberately *not* long simulation experiments. The expensive
environment construction is exercised separately; here, a compact batch orbit
provides enough samples to catch errors in the RK4 plumbing, force-model
selection, and secular J2 behaviour.
"""

import numpy as np
import pytest

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants

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


def propagate_batch(state, duration, steps, *, zonal_J):
    """Propagate through the public batch API without per-step I/O work."""
    return Orbit(
        os0=state,
        end_time=state.J2000 + duration * TimeConstants.sec2cent,
        dt=duration / steps,
        zonal_J=zonal_J,
        verbose=False,
    )


@pytest.fixture(scope="module", params=[
    (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0]), "circular"),
    (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 8.6, 2.0]), "eccentric_inclined"),
])
def two_body_run(request):
    R, V, name = request.param
    start = make_state(R, V)
    start_elements = orbital_elements(R, V)
    period = 2.0 * np.pi * np.sqrt(start_elements["a"] ** 3 / MU)
    # Three periods with 256 RK4 steps per period retain sensitivity to
    # accumulated integration error while remaining cheap through the batch
    # API.  This is substantially shorter than the former 3 x 500-step loop.
    orbit = propagate_batch(start, 3.0 * period, 3 * 256, zonal_J=0)
    finish = orbit.states[orbit.times[-1]]
    return start, start_elements, finish, name


def test_two_body_specific_energy_is_conserved(two_body_run):
    _, initial, finish, name = two_body_run
    final = orbital_elements(finish.R, finish.V)
    relative_drift = abs(final["energy"] - initial["energy"]) / abs(initial["energy"])
    assert relative_drift < 1e-6, f"{name}: specific-energy drift {relative_drift:.2e}"


def test_two_body_angular_momentum_is_conserved(two_body_run):
    _, initial, finish, name = two_body_run
    final = orbital_elements(finish.R, finish.V)
    relative_drift = np.linalg.norm(final["h"] - initial["h"]) / np.linalg.norm(initial["h"])
    assert relative_drift < 1e-7, f"{name}: |h| drift {relative_drift:.2e}"


def test_two_body_eccentricity_vector_is_conserved(two_body_run):
    _, initial, finish, name = two_body_run
    final = orbital_elements(finish.R, finish.V)
    drift = np.linalg.norm(final["e_vec"] - initial["e_vec"])
    assert drift < 2e-6, f"{name}: eccentricity-vector drift {drift:.2e}"


def test_orbit_closes_in_position_after_three_kepler_periods(two_body_run):
    start, _, finish, _ = two_body_run
    assert np.linalg.norm(finish.R - start.R) < 0.25


def test_orbit_closes_in_velocity_after_three_kepler_periods(two_body_run):
    start, _, finish, _ = two_body_run
    assert np.linalg.norm(finish.V - start.V) < 2.5e-4


@pytest.fixture(scope="module")
def j2_secular_run():
    semi_major_axis = RE + 1500.0
    eccentricity = 0.08
    inclination = np.radians(51.6)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    perigee_radius = semi_major_axis * (1.0 - eccentricity)
    perigee_speed = np.sqrt(MU * (2.0 / perigee_radius - 1.0 / semi_major_axis))
    start = make_state(
        np.array([perigee_radius, 0.0, 0.0]),
        np.array([0.0, perigee_speed * np.cos(inclination), perigee_speed * np.sin(inclination)]),
    )
    initial = orbital_elements(start.R, start.V)
    mean_motion = np.sqrt(MU / semi_major_axis**3)
    period = 2.0 * np.pi / mean_motion
    # Retain eight periods for a clean secular signal, but share one compact
    # 96-steps-per-period batch trajectory between the two rate checks.
    orbit = propagate_batch(start, 8.0 * period, 8 * 96, zonal_J=2)
    finish = orbit.states[orbit.times[-1]]
    return initial, orbital_elements(finish.R, finish.V), mean_motion, semi_latus_rectum, inclination, 8.0 * period


def unwrap_rate(end, start, total_time):
    delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
    return delta / total_time


def test_j2_nodal_regression_matches_first_order_analytic_rate(j2_secular_run):
    initial, final, mean_motion, semi_latus_rectum, inclination, total_time = j2_secular_run
    numeric_rate = unwrap_rate(final["raan"], initial["raan"], total_time)
    analytic_rate = -1.5 * mean_motion * J2 * (RE / semi_latus_rectum) ** 2 * np.cos(inclination)
    assert analytic_rate < 0.0
    assert abs(numeric_rate - analytic_rate) / abs(analytic_rate) < 0.05


def test_j2_apsidal_precession_matches_first_order_analytic_rate(j2_secular_run):
    initial, final, mean_motion, semi_latus_rectum, inclination, total_time = j2_secular_run
    numeric_rate = unwrap_rate(final["argp"], initial["argp"], total_time)
    analytic_rate = 0.75 * mean_motion * J2 * (RE / semi_latus_rectum) ** 2 * (5.0 * np.cos(inclination) ** 2 - 1.0)
    assert abs(numeric_rate - analytic_rate) / abs(analytic_rate) < 0.08
