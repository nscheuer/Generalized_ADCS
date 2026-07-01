"""E/I separation via INCLINATION offset -- the SSO-costly trade case.

Companion to suncatcher_ei_oes.py. Both achieve the same RN-plane passive
safety (Eq. 9), but they place the relative inclination vector differently:

    node-based  (suncatcher_ei_oes):  delta-i along (0, 1)  -> delta-i_y only
                                       same inclination, SSO preserved.
    incl-based  (this file):          delta-i along (1, 0)  -> delta-i_x only
                                       spread in inclination, SSO degraded.

Different inclinations precess at different J2 nodal rates, so this formation
shears in RAAN over time and the deputies drift off sun-synchronicity. The
RN-plane safety is identical (the Eq. 9 dot product is axis-agnostic); only the
secular J2 behaviour differs. The analyze report includes the nodal-drift cost.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import RE, ALT_KM, analyze_formation, states_from_roe

A_C_KM = RE + ALT_KM
A_DE_SEP_KM = 0.10            # a_c * de_sep [km]
A_DI_SEP_KM = 0.10            # a_c * di_sep [km]  -- now along delta-i_x (inclination)
M_LADDER = 10                 # N = 2*M + 1 satellites
EPS_KM = 0.050                # minimum safe separation [km] (50 m)


def build_cluster():
    de_sep = A_DE_SEP_KM / A_C_KM
    di_sep = A_DI_SEP_KM / A_C_KM
    ks = [0] + [k for m in range(1, M_LADDER + 1) for k in (m, -m)]

    droe = np.zeros((len(ks), 6))            # [da, dlam, dex, dey, dix, diy]
    for row, k in enumerate(ks):
        droe[row, 2] = k * de_sep            # dex  (delta-e along x, theta = 0)
        droe[row, 4] = k * di_sep            # dix  (delta-i along (1,0): inclination)
    return states_from_roe(droe)


def main():
    R0, V0, meta = build_cluster()
    print(f"E/I separation via INCLINATION: M = {M_LADDER}  (N = {2 * M_LADDER + 1});  "
          f"a_c*de_sep = {A_DE_SEP_KM * 1e3:.0f} m,  a_c*di_sep = {A_DI_SEP_KM * 1e3:.0f} m\n")
    analyze_formation(R0, V0, meta, eps=EPS_KM,
                      label="suncatcher e/i (inclination)", report_sso=True)


if __name__ == "__main__":
    main()
