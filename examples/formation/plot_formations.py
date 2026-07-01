"""Visual comparison of formation geometries: ROE space + relative orbit tracks.

Plots, for each formation, three views:
  1. ROE space      -- where deputies sit in delta-e and delta-i vector space
  2. RT-plane       -- relative orbit projected on radial vs along-track
  3. RN-plane       -- relative orbit projected on radial vs cross-track
The RN-plane is where the dense baseline (delta-i = 0) and the e/i-separation
swarm differ: the dense orbits collapse onto a line through the chief (no
separation perpendicular to flight), while e/i orbits form a tube that avoids
the chief.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formation_lib import RE, ALT_KM, compute_elements, center_index, relative_roe

import suncatcher_hex_oes as dense
import suncatcher_ei_oes as ei

A_C = RE + ALT_KM


def tracks(droe, n=240):
    """Relative orbit over one period: returns dR, dT, dN [km] arrays (N x n)."""
    u = np.linspace(0.0, 2.0 * np.pi, n)
    cu, su = np.cos(u), np.sin(u)
    dR, dT, dN = [], [], []
    for _da, dlam, dex, dey, dix, diy in droe:
        dR.append(-A_C * (dex * cu + dey * su))
        dT.append(A_C * (dlam + 2.0 * dex * su - 2.0 * dey * cu))
        dN.append(A_C * (dix * su - diy * cu))
    return np.array(dR), np.array(dT), np.array(dN)


def roe_of(build):
    R0, V0, meta = build()
    oes = compute_elements(R0, V0)
    c = center_index(meta)
    return relative_roe(oes, c), c


configs = [("Dense (hex, N=91)", dense.build_cluster),
           ("E/I separation (N=21)", ei.build_cluster)]
col_titles = ["ROE space", "relative orbits: RT-plane\n(radial vs along-track)",
              "relative orbits: RN-plane\n(radial vs cross-track)"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for row, (name, build) in enumerate(configs):
    droe, c = roe_of(build)
    dR, dT, dN = tracks(droe)

    # 1. ROE space (delta-e and delta-i vectors), symmetric equal limits
    ax = axes[row, 0]
    ax.scatter(A_C * droe[:, 2], A_C * droe[:, 3], s=22, c="tab:blue", label=r"$a\,\delta e$ (in-plane)")
    ax.scatter(A_C * droe[:, 4], A_C * droe[:, 5], s=30, c="tab:red", marker="^", label=r"$a\,\delta i$ (cross-track)")
    lim = 1.1 * max(A_C * np.hypot(droe[:, 2], droe[:, 3]).max(),
                    A_C * np.hypot(droe[:, 4], droe[:, 5]).max(), 1e-6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.axhline(0, lw=.5, c="k"); ax.axvline(0, lw=.5, c="k"); ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=.3)

    # 2. RT-plane and 3. RN-plane relative orbits, colored by amplitude
    amp = A_C * np.hypot(droe[:, 2], droe[:, 3])
    colors = plt.cm.viridis(amp / max(amp.max(), 1e-9))
    for col, (xx, yy, xlab) in enumerate(
            [(dT, dR, "along-track T [km]"), (dN, dR, "cross-track N [km]")], start=1):
        ax = axes[row, col]
        for i in range(droe.shape[0]):
            ax.plot(xx[i], yy[i], lw=.7, alpha=.7, color=colors[i])
        ax.plot(0, 0, "k*", ms=14)
        ax.set_xlabel(xlab); ax.set_ylabel("radial R [km]")
        ax.set_aspect("equal"); ax.grid(alpha=.3)

    if row == 0:
        for col in range(3):
            axes[0, col].set_title(col_titles[col], fontsize=11)
    fig.text(0.005, 0.74 - 0.5 * row, name, rotation=90, va="center",
             fontsize=12, fontweight="bold")

out = os.path.join(os.path.dirname(__file__), "formation_comparison.png")
fig.tight_layout(rect=(0.02, 0, 1, 1))
fig.savefig(out, dpi=110)
print("wrote", out)
