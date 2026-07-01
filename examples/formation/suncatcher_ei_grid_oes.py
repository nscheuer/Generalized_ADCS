"""E/I separation on a 2-D grid (tilted dense formation).

Extends the 1-D e/i ladder to a full 2-D grid by tilting the dense delta-e
packing: each satellite gets delta-i = c * delta-e (a fixed tilt ratio c, same
for every satellite). Because the tilt is proportional, the RELATIVE vectors
satisfy delta-i_jk = c * delta-e_jk for every pair, so delta-i stays parallel
to delta-e pairwise and Eq. (9) gives RN-plane separation = c * a*|delta-e_jk|
for all pairs -- i.e. c * (the RT-plane pitch). With c = 1 the RN separation
equals the grid pitch.

THE TRADE: delta-i = c*delta-e has a nonzero delta-i_x = c*delta-e_x component
(inclination spread), so this 2-D RN-safe grid is NOT sun-synchronous-preserving
-- the formation shears in RAAN over time (reported below). A true 2-D grid
cannot be simultaneously full-RN-safe AND SSO-preserving; that is exactly why
the pure e/i design (suncatcher_ei_oes) is a 1-D ladder.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import RE, ALT_KM, analyze_formation, states_from_roe, square_disk_amplitudes

A_C_KM = RE + ALT_KM
SEP_KM = 0.10            # grid pitch (a_c*delta-e nearest-neighbor spacing) [km]
RINGS = 5                # square-disk grid: rings=5 -> 81 satellites
C_TILT = 1.0             # delta-i / delta-e ratio; RN separation = C_TILT * pitch
EPS_KM = 0.050           # minimum safe separation [km] (50 m)


def build_cluster():
    # 2-D delta-e grid from the dense square-disk lattice (a_c*delta-e in km)
    u, w = square_disk_amplitudes(rings=RINGS, rho_max=RINGS * SEP_KM)
    dex = -u / A_C_KM            # measured amplitude->ROE map: a*dex = -u, a*dey = +w
    dey = w / A_C_KM

    droe = np.zeros((u.size, 6))     # [da, dlam, dex, dey, dix, diy]
    droe[:, 2] = dex
    droe[:, 3] = dey
    droe[:, 4] = C_TILT * dex        # dix = c*dex  (inclination component -> SSO cost)
    droe[:, 5] = C_TILT * dey        # diy = c*dey  (node component)
    return states_from_roe(droe)


def main():
    R0, V0, meta = build_cluster()
    print(f"E/I grid: square-disk N = {R0.shape[0]};  pitch = {SEP_KM * 1e3:.0f} m;  "
          f"tilt c = {C_TILT}  (RN separation ~ c*pitch = {C_TILT * SEP_KM * 1e3:.0f} m)\n")
    analyze_formation(R0, V0, meta, eps=EPS_KM,
                      label="suncatcher e/i grid", report_sso=True)


if __name__ == "__main__":
    main()
