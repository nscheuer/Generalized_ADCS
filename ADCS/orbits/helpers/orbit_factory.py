import numpy as np
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants, EarthConstants
from ADCS.helpers.math_helpers import normalize

import numpy as np

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize


def create_random_circular_orbit(
    radius_km: float,
    dt: float,
    tf: float,
    use_J2: bool = True,
    fast: bool = False,
) -> Orbit:
    """
    Create a circular orbit with:
      - random position uniformly distributed on the sphere of radius `radius_km`
      - random tangential velocity direction
      - correct circular speed

    Parameters
    ----------
    radius_km : float
        Orbital radius [km].
    dt : float
        Time step [s].
    tf : float
        Final time [s].
    use_J2 : bool
        Whether to include J2 perturbation.
    fast : bool
        Whether to use fast propagation.

    Returns
    -------
    Orbit
        ADCS Orbit object.
    """

    ephem = Ephemeris()

    # --- Gravitational parameter (km^3/s^2) ---
    mu = EarthConstants.mu_e

    # ------------------------------------------------------------------
    # 1) Random position on sphere (uniform)
    # ------------------------------------------------------------------
    r_hat = normalize(np.random.standard_normal(3))   # uniform on S^2
    R = radius_km * r_hat

    # ------------------------------------------------------------------
    # 2) Random tangential direction
    #    Pick random vector, project into tangent plane, normalize
    # ------------------------------------------------------------------
    v_rand = np.random.standard_normal(3)
    v_tan = v_rand - np.dot(v_rand, r_hat) * r_hat    # remove radial component
    v_hat = normalize(v_tan)

    # ------------------------------------------------------------------
    # 3) Circular velocity magnitude
    # ------------------------------------------------------------------
    v_circ = float(np.sqrt(mu / radius_km))            # km/s
    V = v_circ * v_hat

    # ------------------------------------------------------------------
    # 4) Orbital state & Orbit object
    # ------------------------------------------------------------------
    start_j2000_cent = 0.22
    start_time = start_j2000_cent - TimeConstants.sec2cent

    os0 = Orbital_State(
        ephem=ephem,
        J2000=start_time,
        R=R,
        V=V,
    )

    return Orbit(
        os0=os0,
        end_time=start_j2000_cent + tf * TimeConstants.sec2cent,
        dt=dt,
        use_J2=use_J2,
        fast=fast,
        verbose=False,
    )
