"""Suncatcher cluster (centered-hexagonal amplitude lattice, N = 91).

Identical to the square-disk suncatcher config except the amplitude lattice is
a centered hexagon (triangular grid, 60-degree basis). A hexagon of ``rings``
shells holds 1 + 3*rings*(rings+1) points: rings=5 -> N = 91. The footprint
(rho_max = R/2) matches the square-disk cluster, so the two configs differ only
in how the satellites are packed within the same 2:1 elliptical region. All
reusable machinery lives in formation_lib.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

from formation_lib import (
    analyze_formation,
    hcw_cluster,
    hex_amplitudes,
)

RINGS = 5             # 5 -> 91 satellites (centered hexagonal number)
EPS_KM = 0.050        # minimum safe separation [km] (50 m)


def build_cluster():
    u, w = hex_amplitudes(rings=RINGS)
    return hcw_cluster(u, w)


def main():
    R0, V0, meta = build_cluster()
    analyze_formation(R0, V0, meta, eps=EPS_KM, label="suncatcher hex")


if __name__ == "__main__":
    main()
