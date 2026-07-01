"""3-D view of the relative orbits in the chief's RTN frame.

Every bounded relative orbit is a planar 2:1 ellipse. The dense baseline
(delta-i = 0) puts all of them in the orbit plane (N = 0). The e/i-separation
swarm (delta-e parallel delta-i) puts them in the SAME plane yawed about the
radial axis, so the disk acquires cross-track extent -- that tilt is what gives
separation in the RN-plane (robust to along-track uncertainty), which the flat
in-plane dense disk lacks. The translucent grey plane marks the orbit plane.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formation_lib import compute_elements, center_index, relative_roe, relative_track

import suncatcher_hex_oes as dense
import suncatcher_ei_oes as ei


def roe_of(build):
    R0, V0, meta = build()
    oes = compute_elements(R0, V0)
    return relative_roe(oes, center_index(meta))


configs = [("Dense (hex, N=91)\nflat in orbit plane — no cross-track", dense.build_cluster),
           ("E/I separation (N=21)\ndisk yawed ~27° about radial axis", ei.build_cluster)]

fig = plt.figure(figsize=(15, 7))
for col, (name, build) in enumerate(configs):
    droe = roe_of(build)
    dR, dT, dN = relative_track(droe)
    amp = np.hypot(droe[:, 2], droe[:, 3])
    colors = plt.cm.viridis(amp / max(amp.max(), 1e-9))

    ax = fig.add_subplot(1, 2, col + 1, projection="3d")
    lim = 1.05 * max(np.abs(dT).max(), np.abs(dN).max(), np.abs(dR).max(), 1e-6)

    # orbit plane (N = 0) for reference
    gp = np.linspace(-lim, lim, 2)
    PT, PR = np.meshgrid(gp, gp)
    ax.plot_surface(PT, np.zeros_like(PT), PR, color="grey", alpha=0.12,
                    rstride=1, cstride=1, linewidth=0)

    # faint orbit traces ...
    for i in range(droe.shape[0]):
        ax.plot(dT[i], dN[i], dR[i], lw=.5, alpha=.25, color=colors[i])
    # ... with the satellites marked at several phase snapshots around the orbit
    K = 12
    snaps = np.linspace(0, dT.shape[1], K, endpoint=False).astype(int)
    ax.scatter(dT[:, snaps].ravel(), dN[:, snaps].ravel(), dR[:, snaps].ravel(),
               s=9, alpha=.9, depthshade=False,
               color=np.repeat(colors, K, axis=0))
    ax.scatter([0], [0], [0], c="k", marker="*", s=160, depthshade=False)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("along-track T [km]"); ax.set_ylabel("cross-track N [km]")
    ax.set_zlabel("radial R [km]")
    ax.set_title(name, fontsize=12)
    ax.view_init(elev=26, azim=-58)

out = os.path.join(os.path.dirname(__file__), "formation_comparison_3d.png")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
