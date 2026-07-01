"""Trade-space characterization across all formation configs.

Tabulates, for each configuration, the metrics that drive the design trades:
  geometry  -- N, footprint extent (R x T x N)
  collision -- worst-case RN-plane separation (Eq. 9, robust to along-track),
               and closest 3-D approach over an orbit
  SSO/power -- secular nodal shear (RAAN/LTAN divergence; 0 = sun-sync safe)
  ISL       -- median nearest-neighbour range (hop length) and peak range rate
This does not pick a winner; it lays the options side by side.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import (compute_elements, center_index, relative_roe,
                           separation_metrics, sso_drift, link_geometry_stats,
                           connectivity, RE, ALT_KM)

import suncatcher_oes as sq
import suncatcher_hex_oes as hexd
import suncatcher_ei_oes as eilad
import suncatcher_ei_incl_oes as eiinc
import suncatcher_ei_grid_oes as eigrid

A_C = RE + ALT_KM
EPS = 0.050

CONFIGS = [
    ("dense square", sq.build_cluster),
    ("dense hex", hexd.build_cluster),
    ("e/i ladder (node)", eilad.build_cluster),
    ("e/i ladder (incl)", eiinc.build_cluster),
    ("e/i grid (c=1)", eigrid.build_cluster),
]


def characterize(build):
    R0, V0, meta = build()
    oes = compute_elements(R0, V0)
    chief = center_index(meta)
    droe = relative_roe(oes, chief)
    N = R0.shape[0]

    # collision: worst RN over all pairs (robust to along-track), and RN-fail count
    min_rn = np.inf
    for j in range(N):
        for k in range(j + 1, N):
            min_rn = min(min_rn, separation_metrics(droe[k] - droe[j], A_C, EPS)["sep_rn"])

    _, rel = sso_drift(oes, chief)
    shear = np.abs(rel).max() * 365.25 * 86400.0 * A_C        # km/yr
    g = link_geometry_stats(droe)
    return dict(N=N, min_rn=min_rn, min3d=g["min_range"], shear=shear,
                ext=(g["ext_R"], g["ext_T"], g["ext_N"]),
                nn=g["nn_median"], rate=g["max_rate_ms"], diam=g["max_range"])


rows = [(name, characterize(b)) for name, b in CONFIGS]

hdr = (f"{'config':<19}{'N':>4}{'extent R/T/N [km]':>20}{'minRN[m]':>10}"
       f"{'min3D[m]':>10}{'SSOshear':>11}{'ISLhop[km]':>11}{'diam[km]':>9}{'rate[m/s]':>10}")
print(hdr); print("-" * len(hdr))
for name, r in rows:
    ext = "/".join(f"{e:.2f}" for e in r["ext"])
    print(f"{name:<19}{r['N']:>4}{ext:>20}{r['min_rn']*1e3:>10.0f}"
          f"{r['min3d']*1e3:>10.0f}{r['shear']:>9.1f}km/yr{r['nn']:>11.3f}"
          f"{r['diam']:>9.2f}{r['rate']:>10.4f}")

# --- ISL connectivity: node degree (who can you talk to) and routing diameter --- #
# Persistent links only (pair stays within range for the whole orbit).
R_LINKS = [0.25, 0.45, 0.90]   # link-range capabilities [km]
print("\nPersistent-ISL connectivity  (degree = #neighbours; hops = routing diameter):")
print(f"{'config':<19}" + "".join(f"{'R='+str(int(R*1e3))+'m':>22}" for R in R_LINKS))
print(f"{'':<19}" + "".join(f"{'deg(min/med/max) hops':>22}" for _ in R_LINKS))
print("-" * (19 + 22 * len(R_LINKS)))
for name, build in CONFIGS:
    droe = relative_roe(compute_elements(*build()[:2]), 0)
    cells = []
    for R in R_LINKS:
        c = connectivity(droe, R)
        hops = c["diameter"] if c["connected"] else f"x{c['n_components']}"
        cells.append(f"{c['deg_min']}/{c['deg_med']:.0f}/{c['deg_max']} h={hops}".rjust(22))
    print(f"{name:<19}" + "".join(cells))
