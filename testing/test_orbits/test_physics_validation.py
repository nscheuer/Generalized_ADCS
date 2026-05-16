"""
External / analytic validation of the orbit propagator.

The adversarial review found the orbit dynamics physically *correct* but with
NO conservation-law or analytic-element test anywhere: the suite's only
"propagation" check re-ran the same RK4 and compared it to itself. These
tests pin the propagator to textbook celestial mechanics so a future
regression in mu, J2, the sign of the J2 term, or the integrator is caught:

* two-body invariants: specific energy, specific angular-momentum vector,
  and the Laplace-Runge-Lenz (eccentricity) vector are conserved;
* the orbital period matches the Kepler value 2*pi*sqrt(a^3/mu);
* the J2 secular nodal regression and apsidal precession match the
  first-order analytic rates.

All assertions are against independent closed-form references, not the code.
"""

import numpy as np
import pytest

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants

# Thorough multi-orbit propagation; excluded from the fast suite but run in
# full CI (the existing suite uses -m "not slow and not vslow").
pytestmark = pytest.mark.slow

MU = EarthConstants.mu_e          # km^3/s^2
RE = EarthConstants.R_e           # km
J2 = EarthConstants.J2coeff
EPHEM = Ephemeris()


def _state(R, V):
    return Orbital_State(ephem=EPHEM, J2000=0.0,
                         R=np.asarray(R, float), V=np.asarray(V, float))


def _elements(R, V):
    """Classical orbital elements from r, v (km, km/s)."""
    r = np.linalg.norm(R)
    v2 = V @ V
    h = np.cross(R, V)
    hn = np.linalg.norm(h)
    n_vec = np.cross([0, 0, 1.0], h)
    nn = np.linalg.norm(n_vec)
    e_vec = ((v2 - MU / r) * R - (R @ V) * V) / MU
    e = np.linalg.norm(e_vec)
    energy = v2 / 2.0 - MU / r
    a = -MU / (2.0 * energy)
    i = np.arccos(h[2] / hn)
    raan = np.arctan2(h[0], -h[1])  # = atan2(n_y, n_x) for n=[−h_y,h_x,0]
    if nn > 1e-12 and e > 1e-12:    # argp undefined for equatorial/circular
        argp = np.arccos(np.clip((n_vec @ e_vec) / (nn * e), -1, 1))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0
    return dict(a=a, e=e, i=i, raan=raan, argp=argp,
                energy=energy, h=h, e_vec=e_vec)


def _propagate(os0, dt, n, j2=False):
    s = os0
    for _ in range(n):
        s = os0.__class__.propagate_orbit_rk4(s, dt, J2_perturbation_on=j2)
    return s


# --------------------------------------------------------------------------
# Two-body invariants: energy, angular momentum, eccentricity vector.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("R,V,name", [
    (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0]), "circular"),
    (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 8.6, 2.0]), "eccentric-inclined"),
])
def test_two_body_invariants_conserved(R, V, name):
    os0 = _state(R, V)
    el0 = _elements(R, V)
    T = 2 * np.pi * np.sqrt(el0["a"] ** 3 / MU)
    n_per = 500
    n_orbits = 3
    dt = T / n_per

    s = os0
    max_dE = max_dh = max_de = 0.0
    for _ in range(n_orbits * n_per):
        s = os0.__class__.propagate_orbit_rk4(s, dt, J2_perturbation_on=False)
        el = _elements(s.R, s.V)
        max_dE = max(max_dE, abs(el["energy"] - el0["energy"]) / abs(el0["energy"]))
        max_dh = max(max_dh, np.linalg.norm(el["h"] - el0["h"]) / np.linalg.norm(el0["h"]))
        max_de = max(max_de, np.linalg.norm(el["e_vec"] - el0["e_vec"]))
    assert max_dE < 1e-6, f"{name}: specific-energy drift {max_dE:.2e}"
    assert max_dh < 1e-7, f"{name}: |h| drift {max_dh:.2e}"
    assert max_de < 1e-4, f"{name}: eccentricity-vector drift {max_de:.2e}"


# --------------------------------------------------------------------------
# Kepler period: the orbit closes after exactly 2*pi*sqrt(a^3/mu).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("R,V", [
    (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0])),
    (np.array([8000.0, 0.0, 0.0]), np.array([0.0, 7.2, 0.0])),
])
def test_orbit_closes_at_kepler_period(R, V):
    os0 = _state(R, V)
    a = _elements(R, V)["a"]
    T = 2 * np.pi * np.sqrt(a ** 3 / MU)
    n = 1200
    s = _propagate(os0, T / n, n)
    pos_err = np.linalg.norm(s.R - R)
    vel_err = np.linalg.norm(s.V - V)
    # After one Kepler period the state must return to the start; the residual
    # is RK4 truncation only (a wrong period or mu would not close).
    assert pos_err < 1.0, f"position not closed: {pos_err:.3f} km"
    assert vel_err < 1e-3, f"velocity not closed: {vel_err:.2e} km/s"


# --------------------------------------------------------------------------
# J2 secular rates vs the first-order analytic formulae.
# --------------------------------------------------------------------------
def test_j2_nodal_regression_and_apsidal_precession():
    # i = 51.6 deg; a moderately eccentric orbit with perigee well above the
    # atmosphere (a small e makes the argument of perigee ill-defined and
    # dominated by short-period J2 terms ~ 1/e, which contaminates a secular
    # apsidal-rate estimate from osculating endpoints).
    a = RE + 1500.0
    e = 0.08
    i = np.radians(51.6)
    p = a * (1 - e ** 2)
    rp = a * (1 - e)
    vp = np.sqrt(MU * (2.0 / rp - 1.0 / a))
    # Start at perigee on the equator crossing with the right inclination.
    R0 = np.array([rp, 0.0, 0.0])
    V0 = np.array([0.0, vp * np.cos(i), vp * np.sin(i)])
    os0 = _state(R0, V0)

    n_mean = np.sqrt(MU / a ** 3)
    T = 2 * np.pi / n_mean
    raan0 = _elements(R0, V0)["raan"]
    argp0 = _elements(R0, V0)["argp"]

    n_orbits = 8
    steps = 300 * n_orbits
    s = _propagate(os0, n_orbits * T / steps, steps, j2=True)
    el = _elements(s.R, s.V)

    def _unwrap_rate(end, start, total_t):
        d = (end - start + np.pi) % (2 * np.pi) - np.pi
        return d / total_t

    total_t = n_orbits * T
    raan_rate = _unwrap_rate(el["raan"], raan0, total_t)
    argp_rate = _unwrap_rate(el["argp"], argp0, total_t)

    raan_analytic = -1.5 * n_mean * J2 * (RE / p) ** 2 * np.cos(i)
    argp_analytic = 0.75 * n_mean * J2 * (RE / p) ** 2 * (5 * np.cos(i) ** 2 - 1)

    # First-order secular theory vs numerically-propagated osculating mean;
    # a few-percent agreement confirms sign, magnitude and the (Re/p)^2 cos i
    # dependence. (Adversarial-review probe measured ~1% on a similar orbit.)
    assert raan_analytic < 0.0  # prograde i<90deg -> westward nodal regression
    assert abs(raan_rate - raan_analytic) / abs(raan_analytic) < 0.05, \
        f"nodal regression {raan_rate:.3e} vs analytic {raan_analytic:.3e} rad/s"
    assert abs(argp_rate - argp_analytic) / abs(argp_analytic) < 0.08, \
        f"apsidal precession {argp_rate:.3e} vs analytic {argp_analytic:.3e} rad/s"
