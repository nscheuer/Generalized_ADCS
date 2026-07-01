"""E/I vector separation formation (D'Amico's concept, Koenig & D'Amico Sec. 3.3).

Unlike the dense suncatcher (delta-i = 0, safe only in the orbit/RT-plane), the
e/i-separation swarm aligns the relative eccentricity and inclination vectors so
that the dot product delta-e . delta-i guarantees a minimum separation in the
RN-plane (perpendicular to flight) -- the GRACE / TanDEM-X passive-safety trick,
which holds for days even as differential drag drifts delta-lambda.

Per Eq. (24) each deputy j gets integer multipliers Y_j (for delta-e) and Z_j
(for delta-i):

    delta-e_j = Y_j * de_sep * (cos theta, sin theta)
    delta-i_j = Z_j * di_sep * (0, 1)

with Y_j, Z_j nonzero and distinct (Eq. 25). Here we use the diagonal ladder
Y_j = Z_j = j for j = +/-1 .. +/-M (plus the chief at j = 0), and delta-a =
delta-lambda = 0. Eq. (28) gives the minimum phase angle theta that guarantees
eps of RN-plane separation; theta = 90 deg (delta-e parallel to delta-i)
maximizes it. The reusable machinery lives in formation_lib.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import RE, ALT_KM, analyze_formation, states_from_roe

A_C_KM = RE + ALT_KM           # chief semimajor axis [km], for km <-> ROE scaling
A_DE_SEP_KM = 0.10             # a_c * de_sep : relative-eccentricity pitch [km]
A_DI_SEP_KM = 0.10             # a_c * di_sep : relative-inclination pitch [km]
THETA = np.radians(90.0)       # phase of delta-e (delta-i is along the y axis)
M_LADDER = 10                  # ladder half-length -> N = 2*M + 1 satellites
EPS_KM = 0.050                 # minimum safe separation [km] (50 m)


def min_theta(a_de_sep_km, a_di_sep_km, eps_km):
    """Minimum |theta| guaranteeing eps of RN-plane separation (Eq. 28)."""
    de_sep = a_de_sep_km / A_C_KM
    di_sep = a_di_sep_km / A_C_KM
    eps = eps_km / A_C_KM
    s = (eps / (de_sep * di_sep)) * np.sqrt(de_sep**2 + di_sep**2 - eps**2)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def build_cluster():
    de_sep = A_DE_SEP_KM / A_C_KM          # dimensionless ROE pitches
    di_sep = A_DI_SEP_KM / A_C_KM
    ks = [0] + [k for m in range(1, M_LADDER + 1) for k in (m, -m)]

    droe = np.zeros((len(ks), 6))          # [da, dlam, dex, dey, dix, diy]
    for row, k in enumerate(ks):
        droe[row, 2] = k * de_sep * np.cos(THETA)   # dex
        droe[row, 3] = k * de_sep * np.sin(THETA)   # dey
        droe[row, 5] = k * di_sep                   # diy  (delta-i along (0,1))
    return states_from_roe(droe)


def main():
    R0, V0, meta = build_cluster()
    th_min = np.degrees(min_theta(A_DE_SEP_KM, A_DI_SEP_KM, EPS_KM))
    print(f"E/I separation ladder: M = {M_LADDER}  (N = {2 * M_LADDER + 1});  "
          f"theta = {np.degrees(THETA):.1f} deg  (min safe theta = {th_min:.1f} deg)")
    print(f"a_c*de_sep = {A_DE_SEP_KM * 1e3:.0f} m,  "
          f"a_c*di_sep = {A_DI_SEP_KM * 1e3:.0f} m\n")
    analyze_formation(R0, V0, meta, eps=EPS_KM, label="suncatcher e/i separation",
                      report_sso=True)


if __name__ == "__main__":
    main()
