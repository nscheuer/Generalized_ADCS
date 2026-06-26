r"""
Recreate the orbital-dynamics results (Figs. 2 and 3) of

    "Towards a future space-based, highly scalable AI infrastructure system
     design" (Agüera y Arcas et al., 2026, arXiv:2511.19468)

using this repository's orbital dynamics (two-body + J2), exercised exactly as
the formation simulator uses them (``Orbital_State._orbit_dynamics_raw``).

The paper flies an 81-satellite (9x9), ~1 km, planar cluster at 650 km altitude
in free fall (no thrust) under Newtonian gravity + J2. Each satellite is placed
on a bounded Hill-Clohessy-Wiltshire (HCW) relative orbit (the 2:1 along-track:
radial ellipse), so the cluster stays together. We reproduce:

  * Figure 2 -- relative positions vs the central reference satellite S0 in a
    NON-rotating frame fixed to S0's RTN axes at t=0, at 12 phases over one
    orbit (the cluster shape cycles twice per orbit).
  * Figure 3 -- distance from S0 to its 8 nearest neighbours over one orbit.

and check the paper's quantitative statements (bounded cluster, 2:1 geometry,
peripheral satellite altitude +/- R/2, two shape-cycles per orbit, and the small
J2-induced drift relative to a perfectly-closing two-body cluster).

Run:  python examples/formation/recreate_paper_fig2_fig3.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants

MU = EarthConstants.mu_e
RE = EarthConstants.R_e

ALT_KM = 650.0          # mean cluster altitude
R_CLUSTER = 1.0         # cluster radius [km] (along-track half-extent)
INCLINATION = np.radians(98.0)  # sun-synchronous-ish (paper uses dawn-dusk SSO)


# --------------------------------------------------------------------------- #
# Cluster construction: bounded HCW relative orbits on a 9x9 amplitude grid
# --------------------------------------------------------------------------- #
def build_cluster():
    r0 = RE + ALT_KM
    v0 = np.sqrt(MU / r0)
    n = np.sqrt(MU / r0**3)              # mean motion
    period = 2.0 * np.pi / n

    # Reference (S0) circular orbit, inclined; RTN basis at t=0.
    R_ref = r0 * np.array([1.0, 0.0, 0.0])
    V_ref = v0 * np.array([0.0, np.cos(INCLINATION), np.sin(INCLINATION)])
    e_R = R_ref / np.linalg.norm(R_ref)                      # radial (zenith)
    e_N = np.cross(R_ref, V_ref); e_N /= np.linalg.norm(e_N)  # orbit normal
    e_T = np.cross(e_N, e_R)                                  # along-track

    # Amplitude-plane lattice (u, w). A satellite with amplitude params (u, w)
    # follows the drift-free HCW relative ellipse
    #   x_radial(t)    =  u cos(nt) - w sin(nt)
    #   y_alongtrack(t)= -2[u sin(nt) + w cos(nt)]
    # so in the rotating frame it traces the classic 2:1 (along:radial) ellipse,
    # and in the NON-rotating S0 frame it circles a fixed centre at frequency 2n.
    #
    # The paper uses a SQUARE lattice clipped to the cluster disk. Clipping an
    # 11x11 amplitude lattice (spacing rho_max/5) to radius rho_max = R/2 keeps
    # the lattice points with i^2 + j^2 <= 25 -- exactly N(5) = 81 points (the
    # Gauss circle count), matching the paper's 81-satellite cluster. The disk
    # in (u, w) maps to the 2:1 elliptical cluster footprint (+/-R along-track,
    # +/-R/2 radial), so peripheral satellites reach altitude +/- R/2.
    rho_max = R_CLUSTER / 2.0
    half = 5
    spacing = rho_max / half
    idx = np.arange(-half, half + 1)
    II, JJ = np.meshgrid(idx, idx)
    keep = (II**2 + JJ**2) <= half**2 + 1e-9   # disk of radius 5 lattice units
    iu = II[keep]
    iw = JJ[keep]
    u = iu * spacing
    w = iw * spacing

    # Relative state in the rotating RTN frame at t=0.
    x0 = u                       # radial
    y0 = -2.0 * w                # along-track
    xd0 = -w * n                 # radial rate
    yd0 = -2.0 * u * n           # along-track rate (satisfies y_dot = -2 n x: no drift)

    # Map rotating-frame relative state -> inertial relative state
    #   r_rel = x R + y T  ;  v_rel = (xd - n y) R + (yd + n x) T   (omega = n N)
    R0 = np.empty((u.size, 3))
    V0 = np.empty((u.size, 3))
    for i in range(u.size):
        r_rel = x0[i] * e_R + y0[i] * e_T
        v_rel = (xd0[i] - n * y0[i]) * e_R + (yd0[i] + n * x0[i]) * e_T
        R0[i] = R_ref + r_rel
        V0[i] = V_ref + v_rel

    meta = dict(n=n, period=period, e_R=e_R, e_T=e_T, e_N=e_N, u=u, w=w,
                iu=iu, iw=iw, spacing=spacing, r0=r0, V_ref=V_ref, R_ref=R_ref)
    return R0, V0, meta


# --------------------------------------------------------------------------- #
# Propagation with this repo's dynamics (RK4 over all satellites)
# --------------------------------------------------------------------------- #
def propagate(R0, V0, period, n_steps, use_J2=True):
    dyn = Orbital_State._orbit_dynamics_raw
    dt = period / n_steps
    N = R0.shape[0]
    R = R0.copy()
    V = V0.copy()
    R_hist = np.empty((n_steps + 1, N, 3))
    R_hist[0] = R
    for k in range(n_steps):
        for i in range(N):
            r0, v0 = R[i], V[i]
            k1r, k1v = dyn(r0, v0, MU, RE, EarthConstants.J2coeff, use_J2)
            k2r, k2v = dyn(r0 + 0.5 * dt * k1r, v0 + 0.5 * dt * k1v, MU, RE, EarthConstants.J2coeff, use_J2)
            k3r, k3v = dyn(r0 + 0.5 * dt * k2r, v0 + 0.5 * dt * k2v, MU, RE, EarthConstants.J2coeff, use_J2)
            k4r, k4v = dyn(r0 + dt * k3r, v0 + dt * k3v, MU, RE, EarthConstants.J2coeff, use_J2)
            R[i] = r0 + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            V[i] = v0 + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        R_hist[k + 1] = R
    return R_hist


def relative_frame(R_hist, s0_index, e_R, e_T):
    """Relative positions vs S0 in the fixed (t=0) RTN basis. Returns km."""
    rel = R_hist - R_hist[:, s0_index:s0_index + 1, :]
    along = rel @ e_T
    radial = rel @ e_R
    return along, radial  # each (n_steps+1, N)


def rotating_frame_amplitudes(R_hist, s0_index, sat_index, e_N):
    """Radial / along-track amplitudes of one satellite in S0's *rotating*
    (instantaneous) RTN frame -- this is where the bounded HCW orbit is the
    classic 2:1 ellipse. e_N is the (near-constant) orbit normal."""
    radial = np.empty(R_hist.shape[0])
    along = np.empty(R_hist.shape[0])
    for k in range(R_hist.shape[0]):
        R_s0 = R_hist[k, s0_index]
        e_R = R_s0 / np.linalg.norm(R_s0)
        e_T = np.cross(e_N, e_R)
        rel = R_hist[k, sat_index] - R_s0
        radial[k] = rel @ e_R
        along[k] = rel @ e_T
    return np.abs(radial).max(), np.abs(along).max()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(R_hist, meta, s0_index, neighbor_idx, s1_index, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_R, e_T = meta["e_R"], meta["e_T"]
    period = meta["period"]
    n_steps = R_hist.shape[0] - 1
    along, radial = relative_frame(R_hist, s0_index, e_R, e_T)  # km
    along_m, radial_m = along * 1e3, radial * 1e3

    N = R_hist.shape[1]
    is_neighbor = np.zeros(N, bool); is_neighbor[neighbor_idx] = True

    # ---- Figure 2: 12 panels over one orbit, NON-rotating S0 frame ----
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for panel in range(12):
        ax = axes[panel // 4][panel % 4]
        kk = int(round(panel * n_steps / 12))
        # horizontal = negative in-track, vertical = zenith (radial); meters
        hx = -along_m[kk]
        vy = radial_m[kk]
        others = ~is_neighbor
        others[s0_index] = False
        others[s1_index] = False
        ax.scatter(hx[others], vy[others], s=9, c="tab:blue")
        ax.scatter(hx[neighbor_idx], vy[neighbor_idx], s=12, c="magenta")
        ax.scatter([hx[s1_index]], [vy[s1_index]], s=28, c="navy", marker="D")
        ax.scatter([hx[s0_index]], [vy[s0_index]], s=40, c="red")
        # Earth-center direction (-R_hat(t)) in plot coords (-T0, R0)
        eR_now = R_hist[kk, s0_index] / np.linalg.norm(R_hist[kk, s0_index])
        arr = np.array([-(eR_now @ e_T), (eR_now @ e_R)])  # in (-T0, R0) axes... sign:
        arr = np.array([-(-eR_now @ e_T), (-eR_now @ e_R)])  # towards Earth = -eR_now
        ax.annotate("", xy=(arr[0] * 350, arr[1] * 350), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))
        ax.set_title(f"T = {panel} Torbit/12", fontsize=9)
        ax.set_xlim(-1100, 1100); ax.set_ylim(-1100, 1100)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        if panel % 4 == 0:
            ax.set_ylabel("zenith [m]")
        if panel // 4 == 2:
            ax.set_xlabel("-in-track [m]")
    fig.suptitle("Recreation of paper Fig. 2: 81-satellite free-fall cluster "
                 "(two-body + J2), relative to S0 in non-rotating frame", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f2 = os.path.join(outdir, "recreate_fig2.png")
    fig.savefig(f2, dpi=110); plt.close(fig)

    # ---- Figure 3: S0 <-> 8 nearest-neighbour distances over one orbit ----
    t = np.linspace(0.0, 1.0, n_steps + 1)
    dist = np.linalg.norm(R_hist[:, neighbor_idx, :] - R_hist[:, s0_index:s0_index + 1, :], axis=2) * 1e3
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(dist.shape[1]):
        ax.plot(t, dist[:, j], lw=1.2)
    ax.set_xlabel("T / Torbit"); ax.set_ylabel("Distance [m]")
    ax.set_title("Recreation of paper Fig. 3: distance from S0 to its 8 nearest neighbours")
    ax.grid(True, alpha=0.3)
    f3 = os.path.join(outdir, "recreate_fig3.png")
    fig.tight_layout(); fig.savefig(f3, dpi=110); plt.close(fig)

    return f2, f3, dist


# --------------------------------------------------------------------------- #
def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    R0, V0, meta = build_cluster()
    u, w = meta["u"], meta["w"]
    iu, iw = meta["iu"], meta["iw"]
    n_steps = 1200

    # Central satellite S0 (u=w=0).
    s0_index = int(np.argmin(u**2 + w**2))

    # S1 = "maximally distant in in-flight direction at t=0" (most prograde):
    # along-track at t=0 is -2w, so the most prograde satellite has min w.
    along0 = -2.0 * w
    s1_index = int(np.argmax(along0))

    # 8-neighbourhood of S0: grid-adjacent lattice cells (Moore neighbourhood),
    # i.e. |di| <= 1 and |dj| <= 1 (excluding S0) -- the paper's definition.
    di = iu - iu[s0_index]
    dj = iw - iw[s0_index]
    neighbor_idx = np.where((np.maximum(np.abs(di), np.abs(dj)) == 1))[0]

    # Propagate one orbit with J2 (the paper's model) and, for reference, two-body.
    R_j2 = propagate(R0, V0, meta["period"], n_steps, use_J2=True)
    R_2body = propagate(R0, V0, meta["period"], n_steps, use_J2=False)

    f2, f3, dist = make_figures(R_j2, meta, s0_index, neighbor_idx, s1_index, outdir)

    # --- Quantitative checks against the paper ---
    print("=" * 70)
    print("Recreating Agüera y Arcas et al. (2026) orbital Figs. 2 & 3")
    print("=" * 70)
    print(f"{R0.shape[0]}-sat planar cluster (square lattice clipped to disk), "
          f"alt={ALT_KM} km, R={R_CLUSTER} km, spacing={meta['spacing']*1e3:.0f} m,")
    print(f"period={meta['period']/60:.1f} min, i={np.degrees(INCLINATION):.0f} deg")

    # Bounded: two-body cluster closes after one orbit; J2 adds a small drift.
    rel_2body = R_2body - R_2body[:, s0_index:s0_index + 1, :]
    rel_j2 = R_j2 - R_j2[:, s0_index:s0_index + 1, :]
    close_2body = np.linalg.norm(rel_2body[-1] - rel_2body[0], axis=1).max() * 1e3
    close_j2 = np.linalg.norm(rel_j2[-1] - rel_j2[0], axis=1).max() * 1e3
    print(f"\nBounded relative motion (max |r_rel(T) - r_rel(0)|):")
    print(f"  two-body : {close_2body:8.2f} m   (should be ~0: cluster closes)")
    print(f"  with J2  : {close_j2:8.2f} m   (small J2-induced drift per orbit)")

    # Cluster stays inside radius R.
    max_extent = np.linalg.norm(rel_j2, axis=2).max() * 1e3
    print(f"\nMax cluster extent from S0 over the orbit: {max_extent:6.1f} m "
          f"(bounded within R = {R_CLUSTER*1e3:.0f} m)")

    # Peripheral satellite S1: geocentric-altitude excursion ~ +/- R/2 (the
    # paper's "apoapsis at a + R/2, periapsis at a - R/2"), and the 2:1
    # along-track:radial ellipse in the rotating frame.
    # Altitude relative to S0 (removes the reference orbit's own ~km-scale J2
    # radial breathing, which is common to the whole cluster).
    alt_s1 = (np.linalg.norm(R_j2[:, s1_index, :], axis=1)
              - np.linalg.norm(R_j2[:, s0_index, :], axis=1)) * 1e3
    rad_amp, along_amp = rotating_frame_amplitudes(R_j2, s0_index, s1_index, meta["e_N"])
    print(f"\nPeripheral satellite S1 (paper: altitude +/-R/2, rotating-frame 2:1 ellipse):")
    print(f"  altitude relative to S0: {alt_s1.min():+7.1f} .. {alt_s1.max():+7.1f} m "
          f"(paper: +/-{R_CLUSTER*1e3/2:.0f} m)")
    print(f"  rotating-frame radial amplitude    : {rad_amp*1e3:6.1f} m  (paper +/-R/2 = {R_CLUSTER*1e3/2:.0f})")
    print(f"  rotating-frame along-track amplitude: {along_amp*1e3:6.1f} m  (paper +/-R = {R_CLUSTER*1e3:.0f})")
    print(f"  -> along:radial ratio = {along_amp/rad_amp:.3f} : 1  (paper 2:1)")

    # Two shape-cycles per orbit: S1's distance to S0 has period ~ T/2.
    s1_dist = np.linalg.norm(rel_j2[:, s1_index, :], axis=1)
    # count maxima
    peaks = np.sum((s1_dist[1:-1] > s1_dist[:-2]) & (s1_dist[1:-1] > s1_dist[2:]))
    print(f"\nShape cycles per orbit (S1 distance maxima): {peaks}  (paper: 2)")

    # Nearest-neighbour distance range (paper Fig 3: ~100-275 m).
    print(f"\nNearest-neighbour distance range over one orbit (paper ~100-275 m):")
    print(f"  min = {dist.min():6.1f} m   max = {dist.max():6.1f} m")

    print(f"\nSaved figures:\n  {f2}\n  {f3}")


if __name__ == "__main__":
    main()
