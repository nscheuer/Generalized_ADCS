"""Suncatcher cluster (square-disk amplitude lattice, N = 81).

A (2*rings+1)^2 amplitude lattice clipped to the cluster disk keeps the
Gauss-circle count of points: rings=5 -> N(5) = 81, matching the paper's
81-satellite cluster. The disk in (u, w) maps to the 2:1 elliptical cluster
footprint (+/-R along-track, +/-R/2 radial). All reusable machinery lives in
formation_lib.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

from formation_lib import (
    analyze_formation,
    hcw_cluster,
    square_disk_amplitudes,
)

RINGS = 5             # 5 -> 81 satellites
EPS_KM = 0.050        # minimum safe separation [km] (50 m)


def build_cluster():
    u, w = square_disk_amplitudes(rings=RINGS)
    return hcw_cluster(u, w)


def main():
    R0, V0, meta = build_cluster()
    analyze_formation(R0, V0, meta, eps=EPS_KM, label="suncatcher square-disk")


if __name__ == "__main__":
    main()
