import numpy as np
import pytest

from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


EPHEM = Ephemeris()
J2000_TT_JD = 2451545.0


def make_orbital_state(j2000: float = 0.22, R=None, V=None) -> Orbital_State:
    return Orbital_State(
        ephem=EPHEM,
        J2000=j2000,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else R,
        V=np.array([0.0, 7.5, 0.0]) if V is None else V,
        B=np.array([2e-5, -1e-5, 3e-5]),
    )


def build_orbit(dt_s: float = 300.0, n: int = 12):
    start = make_orbital_state(j2000=0.0, R=np.array([6878.0, 0.0, 0.0]), V=np.array([0.0, 7.613, 0.0]))
    states = [start]
    for _ in range(n):
        states.append(states[-1].propagate_orbit_rk4(dt_s))
    return Orbit(states), dt_s


def test_ecef_to_geocentric_roundtrip_is_identity():
    orbital_state = make_orbital_state()
    rng = np.random.default_rng(0)
    for _ in range(25):
        vector = rng.normal(size=3)
        assert np.allclose(orbital_state.ecef_to_geocentric(orbital_state.geocentric_to_ecef(vector)), vector, atol=1e-12)


def test_geocentric_to_ecef_roundtrip_is_identity():
    orbital_state = make_orbital_state()
    rng = np.random.default_rng(0)
    for _ in range(25):
        vector = rng.normal(size=3)
        assert np.allclose(orbital_state.geocentric_to_ecef(orbital_state.ecef_to_geocentric(vector)), vector, atol=1e-12)


def test_ecef_geocentric_transform_matrix_is_orthonormal():
    transform = make_orbital_state()._ecef_to_geo
    assert np.allclose(transform @ transform.T, np.eye(3), atol=1e-12)


def test_ecef_geocentric_transform_matrix_is_right_handed():
    transform = make_orbital_state()._ecef_to_geo
    assert np.isclose(np.linalg.det(transform), 1.0, atol=1e-9)


def test_j2000_zero_uses_tt_epoch_not_tai():
    timescale = EPHEM.ts
    sun_position = lambda time: np.asarray(EPHEM.earth.at(time).observe(EPHEM.sun).apparent().position.km, dtype=float)
    reference_tt = sun_position(timescale.tt_jd(J2000_TT_JD))
    reference_tai = sun_position(timescale.tai_jd(J2000_TT_JD))
    orbital_state = make_orbital_state(j2000=0.0)
    sun = orbital_state.get_sun_eci()

    assert np.allclose(sun, reference_tt, rtol=0.0, atol=1e-3)
    assert np.linalg.norm(reference_tt - reference_tai) > 100.0
    assert np.linalg.norm(sun - reference_tai) > 100.0


def test_density_interpolation_matches_table_nodes():
    model = DensityModel()
    for altitude, density in zip(model.altitude_range, model.rho_range):
        assert np.isclose(model.interpolate(float(altitude)), float(density), rtol=1e-9)


def test_density_interpolation_is_log_linear_between_nodes():
    model = DensityModel()
    index = len(model.altitude_range) // 2
    midpoint = 0.5 * (model.altitude_range[index] + model.altitude_range[index + 1])
    geometric = np.sqrt(model.rho_range[index] * model.rho_range[index + 1])
    linear = 0.5 * (model.rho_range[index] + model.rho_range[index + 1])
    interpolated = model.interpolate(midpoint)
    assert np.isclose(interpolated, geometric, rtol=1e-9)
    assert abs(np.log(interpolated) - np.log(geometric)) < abs(np.log(linear) - np.log(geometric))


def test_density_interpolation_is_monotonic_decreasing():
    model = DensityModel()
    sweep = np.linspace(model.altitude_range[0] + 1.0, model.altitude_range[-1] - 1.0, 200)
    densities = np.array([model.interpolate(altitude) for altitude in sweep])
    assert np.all(np.diff(densities) <= 1e-30)


def test_density_extrapolation_decays_above_table():
    model = DensityModel()
    top = float(model.altitude_range[-1])
    assert model.interpolate(top + 5000.0) < model.interpolate(top) * 1e-3
    assert model.interpolate(1.0e6) < model.interpolate(top) * 1e-12


def test_density_interpolation_clamps_below_table():
    model = DensityModel()
    assert model.interpolate(float(model.altitude_range[0]) - 50.0) == pytest.approx(float(model.rho_range[0]))
    assert np.isfinite(model.interpolate(-100.0))


def test_get_os_returns_exact_node_state():
    orbit, _ = build_orbit()
    t1 = orbit.times[1]
    node = orbit.get_os(float(t1))
    assert np.allclose(node.R, orbit.states[orbit.times[1]].R, atol=1e-9)


def test_get_os_midpoint_matches_fine_rk4_truth():
    orbit, dt_s = build_orbit()
    t0 = orbit.times[0]
    t1 = orbit.times[1]
    midpoint = 0.5 * (t0 + t1)
    truth = orbit.states[t0]
    for _ in range(400):
        truth = truth.propagate_orbit_rk4(0.5 * dt_s / 400)
    interpolated = orbit.get_os(float(midpoint))
    assert np.linalg.norm(interpolated.R - truth.R) < 1.0


def test_get_os_midpoint_recomputes_orthonormal_frames():
    orbit, _ = build_orbit()
    midpoint = 0.5 * (orbit.times[0] + orbit.times[1])
    interpolated = orbit.get_os(float(midpoint))
    for matrix in (interpolated._R_eci2ecef, interpolated.ECI2ENUmat):
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        assert np.isclose(abs(np.linalg.det(matrix)), 1.0, atol=1e-6)
