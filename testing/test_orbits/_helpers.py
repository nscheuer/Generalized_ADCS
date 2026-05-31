import numpy as np

from ADCS.helpers.math_helpers import normalize, random_n_unit_vec
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants


def make_reference_orbital_state():
    ephem = Ephemeris()
    zero_time = 0.22
    r = np.array([7078.137, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(EarthConstants.mu_e / np.linalg.norm(r)), 0.0])
    return Orbital_State(ephem=ephem, J2000=zero_time, R=r, V=v, fast=True)


def make_random_orbital_state(seed=123):
    rng = np.random.default_rng(seed)
    n_n = normalize(rng.standard_normal(3))
    ecc = rng.uniform(0.1, 0.4)
    rp = rng.uniform(6800.0, 9000.0)
    a = rp / (1.0 - ecc)
    vp = np.sqrt(EarthConstants.mu_e * (1.0 + ecc) / (1.0 - ecc) / a)
    h = rp * vp
    th = rng.uniform(0.0, 2.0 * np.pi)
    r = h * h / (EarthConstants.mu_e * (1.0 + ecc * np.cos(th)))
    v = np.sqrt(EarthConstants.mu_e * (2.0 / r - 1.0 / a))

    n_rp = normalize(np.cross(rng.standard_normal(3), n_n))
    n_vp = normalize(np.cross(n_n, n_rp))

    rvec = r * (n_rp * np.cos(th) + n_vp * np.sin(th))
    cphi = (1.0 + ecc * np.cos(th)) / np.sqrt(1.0 + ecc * ecc + 2.0 * ecc * np.cos(th))
    phi = np.arccos(cphi) * np.sign(np.sin(th))
    psi = th + 0.5 * np.pi - phi
    vvec = v * (n_rp * np.cos(psi) + n_vp * np.sin(psi))

    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=rvec, V=vvec, fast=True)


def make_orbit_family(dt=3600.0, n_steps=24 * 2):
    os0 = make_reference_orbital_state()
    end_time = os0.J2000 + TimeConstants.sec2cent * dt * n_steps

    orb0 = Orbit(os0=os0, fast=True, verbose=False)
    orb1 = Orbit(os0=os0, end_time=end_time, dt=dt, fast=False, verbose=False)
    orb2 = Orbit([*orb1.states.values()])
    orb3 = Orbit(os0=os0, end_time=end_time, dt=dt, fast=True, verbose=False)
    return os0, orb0, orb1, orb2, orb3, dt, n_steps, end_time
