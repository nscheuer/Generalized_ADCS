__all__ = ["derive_environment", "j2000_to_jd_tt"]

from typing import Dict

import numpy as np
from skyfield import framelib

from ADCS.helpers.math_helpers import dcm_to_quat
from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.helpers.animations.godot.interp import quat_continuity


def j2000_to_jd_tt(j2000: np.ndarray | float) -> np.ndarray | float:
    r"""
    Convert the framework's time variable to a TT Julian date.

    ``J2000`` is Julian centuries of TT since J2000.0 (JD 2451545.0 TT), the
    same convention as :meth:`~ADCS.orbits.orbital_state.Orbital_State.j2000_to_tai`.
    """
    return np.asarray(j2000, dtype=float) * 36525.0 + 2451545.0


def _as_rows(xyz: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(xyz, dtype=float)
    return a.T if a.shape == (3, n) else a.reshape(n, 3)


def derive_environment(ephem: Ephemeris, jd_tt: np.ndarray) -> Dict[str, np.ndarray]:
    r"""
    Sun, Moon and Earth orientation at many epochs in one vectorized pass.

    Uses the same Skyfield calls as :class:`~ADCS.orbits.orbit.Orbit`, so the
    Sun vector matches ``Orbital_State.S``: apparent geocentric position
    (light time and aberration applied) on GCRS axes.

    :param ephem: Ephemeris providing the kernel and timescale.
    :param jd_tt: TT Julian dates, shape ``(N,)``.
    :return: ``r_sun`` and ``r_moon`` in km ``(N, 3)``; ``R_itrs_gcrs``
        ``(N, 3, 3)`` with ``v_itrs = R @ v_gcrs``; ``q_itrs_gcrs`` ``(N, 4)``,
        the same rotation as a continuous scalar-first quaternion.
    """
    jd_tt = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    n = jd_tt.size
    t = ephem.ts.tt_jd(jd_tt)
    earth = ephem.earth.at(t)

    r_sun = _as_rows(earth.observe(ephem.sun).apparent().position.km, n)
    r_moon = _as_rows(earth.observe(ephem.planets["moon"]).apparent().position.km, n)

    R = np.asarray(framelib.itrs.rotation_at(t), dtype=float)
    R = np.transpose(R, (2, 0, 1)) if R.shape == (3, 3, n) else R.reshape(n, 3, 3)
    q = quat_continuity(np.array([dcm_to_quat(np.ascontiguousarray(Ri)) for Ri in R]))

    return {"r_sun": r_sun, "r_moon": r_moon, "R_itrs_gcrs": R, "q_itrs_gcrs": q}
