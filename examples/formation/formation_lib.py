"""Reusable building blocks for HCW amplitude-lattice formation studies.

This module factors out everything that is independent of the particular
amplitude lattice (square disk, centered hexagon, ...) so that a concrete
configuration script only has to:

    1. choose an amplitude lattice  -> (u, w) arrays
    2. build the cluster           -> hcw_cluster(u, w)
    3. analyze it                  -> analyze_formation(R0, V0, meta, ...)

The relative-orbital-element (ROE) separation metrics implement Eqs. (9),
(13) and (14) of Koenig & D'Amico, "Robust and Safe N-Spacecraft Swarming in
Perturbed Near-Circular Orbits" (ISSFD 2017).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants

MU = EarthConstants.mu_e
RE = EarthConstants.R_e
J2 = EarthConstants.J2 / (MU * RE**2)   # dimensionless J2 (EarthConstants.J2 is J2*mu*Re^2)
SSO_RATE = 2.0 * np.pi / (365.25 * 86400.0)   # nodal rate for sun-synchronicity [rad/s]

# Default suncatcher scenario parameters (overridable per call).
ALT_KM = 650.0                   # mean cluster altitude [km]
R_CLUSTER = 1.0                  # cluster radius [km] (along-track half-extent)
INCLINATION = np.radians(98.0)   # sun-synchronous-ish (paper uses dawn-dusk SSO)

# Column layout of the per-satellite element table built by compute_elements().
A, INC, OMEGA, AOP, NU, MM, ECC, EA, MA, LAMB, EX, EY, IX, IY = range(14)


# --------------------------------------------------------------------------- #
# Amplitude lattices (u, w). Each generator returns the amplitude-plane points
# that seed the drift-free HCW relative ellipses; swap the generator to change
# the cluster geometry without touching anything downstream.
# --------------------------------------------------------------------------- #
def square_disk_amplitudes(rings=5, rho_max=R_CLUSTER / 2.0):
    """Square lattice clipped to a disk -> Gauss-circle count of points.

    Clipping a (2*rings+1)^2 lattice (spacing rho_max/rings) to radius rho_max
    keeps the points with i^2 + j^2 <= rings^2. For rings=5 that is N(5) = 81
    points, matching the paper's 81-satellite cluster.
    """
    spacing = rho_max / rings
    idx = np.arange(-rings, rings + 1)
    ii, jj = np.meshgrid(idx, idx)
    keep = (ii**2 + jj**2) <= rings**2 + 1e-9
    return ii[keep] * spacing, jj[keep] * spacing


def hex_amplitudes(rings=5, rho_max=R_CLUSTER / 2.0):
    """Centered hexagonal (triangular) lattice -> centered-hexagonal count.

    A hexagon of ``rings`` shells holds 1 + 3*rings*(rings+1) points; for
    rings=5 that is exactly 91. Points lie on a triangular grid (60-degree
    basis) of pitch ``rho_max/rings`` so the footprint matches the square-disk
    cluster of the same rho_max.
    """
    spacing = rho_max / rings
    u, w = [], []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            s = -q - r                       # cube coordinate; hex if max(|q|,|r|,|s|) <= rings
            if max(abs(q), abs(r), abs(s)) <= rings:
                u.append(spacing * (q + r / 2.0))
                w.append(spacing * (np.sqrt(3.0) / 2.0 * r))
    return np.array(u), np.array(w)


# --------------------------------------------------------------------------- #
# Cluster construction: amplitude lattice -> bounded HCW relative orbits.
# --------------------------------------------------------------------------- #
def hcw_cluster(u, w, alt_km=ALT_KM, inclination=INCLINATION):
    """Build inertial states for a drift-free HCW cluster from amplitudes.

    A satellite with amplitude params (u, w) follows the bounded relative orbit
        x_radial(t)     =  u cos(nt) - w sin(nt)
        y_alongtrack(t) = -2[u sin(nt) + w cos(nt)]
    which traces the classic 2:1 (along:radial) ellipse in the rotating frame
    and is free of secular along-track drift (y_dot = -2 n x at t=0).
    """
    r0 = RE + alt_km
    v0 = np.sqrt(MU / r0)
    n = np.sqrt(MU / r0**3)
    period = 2.0 * np.pi / n

    # Reference (S0) circular orbit, inclined; RTN basis at t=0.
    R_ref = r0 * np.array([1.0, 0.0, 0.0])
    V_ref = v0 * np.array([0.0, np.cos(inclination), np.sin(inclination)])
    e_R = R_ref / np.linalg.norm(R_ref)                       # radial (zenith)
    e_N = np.cross(R_ref, V_ref); e_N /= np.linalg.norm(e_N)   # orbit normal
    e_T = np.cross(e_N, e_R)                                   # along-track

    # Relative state in the rotating RTN frame at t=0.
    x0 = u                       # radial
    y0 = -2.0 * w                # along-track
    xd0 = -w * n                 # radial rate
    yd0 = -2.0 * u * n           # along-track rate (no drift)

    # Map rotating-frame relative state -> inertial relative state
    #   r_rel = x R + y T ;  v_rel = (xd - n y) R + (yd + n x) T   (omega = n N)
    R0 = np.empty((u.size, 3))
    V0 = np.empty((u.size, 3))
    for i in range(u.size):
        r_rel = x0[i] * e_R + y0[i] * e_T
        v_rel = (xd0[i] - n * y0[i]) * e_R + (yd0[i] + n * x0[i]) * e_T
        R0[i] = R_ref + r_rel
        V0[i] = V_ref + v_rel

    meta = dict(n=n, period=period, e_R=e_R, e_T=e_T, e_N=e_N, u=u, w=w,
                r0=r0, V_ref=V_ref, R_ref=R_ref)
    return R0, V0, meta


def _oe_to_rv(a, e, inc, Omega, omega, M):
    """Classical orbital elements -> inertial position/velocity (Kepler solve)."""
    E = M if e < 0.8 else np.pi
    for _ in range(60):                      # Newton iteration on Kepler's equation
        E -= (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0),
                          np.sqrt(1 - e) * np.cos(E / 2.0))
    r = a * (1.0 - e * np.cos(E))
    p = a * (1.0 - e**2)
    r_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(MU / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(omega), np.sin(omega)
    Q = np.array([                            # perifocal -> ECI (Rz(O) Rx(i) Rz(w))
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [si * sw,                 si * cw,                  ci]])
    return Q @ r_pf, Q @ v_pf


def states_from_roe(droe, alt_km=ALT_KM, inclination=INCLINATION):
    """Build inertial states from quasi-nonsingular ROE relative to a chief.

    ``droe`` is an (N, 6) array of [da, dlam, dex, dey, dix, diy] for each
    satellite (the chief is whichever row is all zeros). This is the exact
    inverse of compute_elements()+relative_roe(): each ROE row is converted to
    absolute Keplerian elements (relative to a circular chief at the ascending
    node, u_c = 0) and then to a state vector, so the round trip is identity.

    Returns (R0, V0, meta), where meta carries the ROE set so downstream tools
    (center_index, analyze_formation) work without an amplitude lattice.
    """
    droe = np.atleast_2d(np.asarray(droe, dtype=float))
    r0 = RE + alt_km
    n = np.sqrt(MU / r0**3)
    period = 2.0 * np.pi / n
    a_c, i_c = r0, inclination          # chief: circular, omega_c = Omega_c = M_c = 0

    R0 = np.empty((droe.shape[0], 3))
    V0 = np.empty((droe.shape[0], 3))
    for k, (da, dlam, dex, dey, dix, diy) in enumerate(droe):
        a_d = a_c * (1.0 + da)
        e_d = np.hypot(dex, dey)                       # chief e_c = 0
        omega_d = np.arctan2(dey, dex) if e_d > 0 else 0.0
        i_d = i_c + dix
        dOmega = diy / np.sin(i_c)
        Omega_d = dOmega
        M_d = dlam - omega_d - np.cos(i_c) * dOmega    # invert dlam = M + w + cos(i)*Omega
        R0[k], V0[k] = _oe_to_rv(a_d, e_d, i_d, Omega_d, omega_d, M_d)

    R_ref = r0 * np.array([1.0, 0.0, 0.0])
    V_ref = np.sqrt(MU / r0) * np.array([0.0, np.cos(i_c), np.sin(i_c)])
    e_R = R_ref / np.linalg.norm(R_ref)
    e_N = np.cross(R_ref, V_ref); e_N /= np.linalg.norm(e_N)
    e_T = np.cross(e_N, e_R)
    meta = dict(n=n, period=period, e_R=e_R, e_T=e_T, e_N=e_N, roe=droe,
                r0=r0, V_ref=V_ref, R_ref=R_ref)
    return R0, V0, meta


def center_index(meta):
    """Index of the central (zero-amplitude / zero-ROE) chief satellite."""
    if "u" in meta and "w" in meta:
        return int(np.argmin(meta["u"]**2 + meta["w"]**2))
    if "roe" in meta:
        return int(np.argmin((meta["roe"]**2).sum(axis=1)))
    return 0


# --------------------------------------------------------------------------- #
# Osculating Keplerian elements from inertial state (singularity-robust).
# --------------------------------------------------------------------------- #
def compute_elements(R0, V0, e_tol=1e-11, n_tol=1e-11):
    """Return an (N, 14) table of Keplerian + equinoctial elements per row.

    Columns: [a, inc, Omega, omega, nu, n, e, E, M, lamb, ex, ey, ix, iy]
    (see the module-level A, INC, ... index constants). Near-circular and
    near-equatorial singularities follow the usual Vallado conventions: the
    undefined angle is set to 0 and the argument of latitude / true longitude
    is substituted so the result is always finite.
    """
    def _angle(cos_val, flip):
        ang = np.arccos(np.clip(cos_val, -1.0, 1.0))
        return 2 * np.pi - ang if flip else ang

    oes = np.empty((R0.shape[0], 14))
    for i in range(R0.shape[0]):
        r, v = R0[i], V0[i]
        r_norm = np.linalg.norm(r)
        hvec = np.cross(r, v)
        evec = np.cross(v, hvec) / MU - r / r_norm
        nvec = np.cross([0, 0, 1], hvec)

        h_norm = np.linalg.norm(hvec)
        n_norm = np.linalg.norm(nvec)
        e = np.linalg.norm(evec)

        a = 1 / (2 / r_norm - np.linalg.norm(v)**2 / MU)
        inc = np.arccos(np.clip(hvec[2] / h_norm, -1.0, 1.0))
        n = np.sqrt(MU / a**3)

        equatorial = n_norm < n_tol
        circular = e < e_tol

        Omega = 0.0 if equatorial else _angle(nvec[0] / n_norm, nvec[1] < 0)

        if circular:
            omega = 0.0
            if equatorial:
                nu = _angle(r[0] / r_norm, r[1] < 0)                       # true longitude
            else:
                nu = _angle(np.dot(nvec, r) / (n_norm * r_norm), r[2] < 0)  # arg of latitude
        else:
            nu = _angle(np.dot(evec, r) / (e * r_norm), np.dot(r, v) < 0)
            if equatorial:
                omega = _angle(evec[0] / e, evec[1] < 0)                   # true longitude of periapsis
            else:
                omega = _angle(np.dot(nvec, evec) / (n_norm * e), evec[2] < 0)

        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        M = E - e * np.sin(E)
        lamb = nu + omega + Omega * np.cos(inc)
        oes[i] = [a, inc, Omega, omega, nu, n, e, E, M, lamb,
                  e * np.cos(omega), e * np.sin(omega), inc, Omega * np.sin(inc)]
    return oes


# --------------------------------------------------------------------------- #
# Relative orbital elements and the Koenig & D'Amico separation metrics.
# --------------------------------------------------------------------------- #
def _wrap(x):
    """Wrap angle(s) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def relative_roe(oes, chief):
    """Quasi-nonsingular ROE of every satellite relative to ``chief`` (Eq. 1).

    Returns an (N, 6) array of [da, dlam, dex, dey, dix, diy]. Each angle
    DIFFERENCE is wrapped to (-pi, pi]; for near-circular orbits omega and M
    are individually ill-defined, so dlam is wrapped as a whole (only their
    sum, the mean longitude, is meaningful).
    """
    a_c = oes[chief, A]
    i_c = oes[chief, INC]
    d_Omega = _wrap(oes[:, OMEGA] - oes[chief, OMEGA])
    d_oes = np.empty((oes.shape[0], 6))
    d_oes[:, 0] = (oes[:, A] - a_c) / a_c                                  # da
    d_oes[:, 1] = _wrap(_wrap(oes[:, MA] - oes[chief, MA])
                        + _wrap(oes[:, AOP] - oes[chief, AOP])
                        + np.cos(i_c) * d_Omega)                           # dlam
    d_oes[:, 2] = oes[:, EX] - oes[chief, EX]                              # dex
    d_oes[:, 3] = oes[:, EY] - oes[chief, EY]                              # dey
    d_oes[:, 4] = _wrap(oes[:, INC] - i_c)                                 # dix
    d_oes[:, 5] = np.sin(i_c) * d_Omega                                    # diy
    return d_oes


def separation_metrics(droe, a_c, eps):
    """Separation metrics for a deputy with ROE ``droe`` w.r.t. a chief.

    ``droe`` is the quasi-nonsingular ROE [da, dlam, dex, dey, dix, diy].
    ``a_c`` is the chief semimajor axis and ``eps`` the minimum safe
    separation, both in km. Returns a dict with:

      sep_rn   Eq. (9):  minimum separation in the RN-plane (cross-track /
               radial), in km. Driven by the relative incl. vector; 0 when
               delta-i is parallel to delta-e (e.g. a purely in-plane cluster).
      sep_rt   Eq. (13): minimum RT-plane (radial / along-track) separation for
               a deputy that does NOT encircle the chief, a_c|dlam| - 2 a_c|de|,
               in km. Negative => the deputy encircles the chief (use Eq. 14).
      f14      Eq. (14): the largest a_c|dlam| that still guarantees eps of
               clearance for an encircling deputy, in km.
      aclam    a_c|dlam| (the along-track quantity Eq. 14 bounds), in km.
      margin14 f14 - aclam; >= 0 iff Eq. (14) is satisfied.
      ok9/ok13/ok14  whether each constraint holds at the given eps.
    """
    _da, dlam, dex, dey, dix, diy = droe
    de = np.hypot(dex, dey)                       # |delta e|
    di = np.hypot(dix, diy)                       # |delta i|
    dedi = dex * dix + dey * diy                  # delta e . delta i
    de_plus = np.hypot(dex + dix, dey + diy)      # |delta e + delta i|
    de_minus = np.hypot(dex - dix, dey - diy)     # |delta e - delta i|

    # Eq. (9): minimum RN-plane separation.
    denom = np.sqrt(de**2 + di**2 + de_plus * de_minus)
    sep_rn = 0.0 if denom == 0.0 else a_c * np.sqrt(2.0) * abs(dedi) / denom

    # Eq. (13): minimum RT-plane separation, non-encircling deputy.
    sep_rt = a_c * abs(dlam) - 2.0 * a_c * de

    # Eq. (14): safe along-track bound for an encircling deputy.
    acde = a_c * de
    aclam = a_c * abs(dlam)
    if acde >= 2.0 * eps:
        f14 = 2.0 * acde - eps
    elif acde >= eps:
        f14 = np.sqrt(3.0 * (acde**2 - eps**2))
    else:
        f14 = 0.0   # a_c|de| < eps: ellipse smaller than the safety ball
    margin14 = f14 - aclam

    return dict(sep_rn=sep_rn, sep_rt=sep_rt, f14=f14, aclam=aclam,
                margin14=margin14,
                ok9=sep_rn >= eps, ok13=sep_rt >= eps, ok14=aclam <= f14)


# --------------------------------------------------------------------------- #
# J2 sun-synchronicity diagnostic
# --------------------------------------------------------------------------- #
def relative_track(droe, n=240, alt_km=ALT_KM):
    """Relative orbit over one period in the chief's RTN frame (da assumed 0).

    ``droe`` is an (N, 6) ROE array; samples argument of latitude over [0, 2pi).
    Returns (dR, dT, dN), each (N, n) in km: radial, along-track, cross-track
    relative position. Uses the standard circular-chief ROE -> RTN map.
    """
    a_c = RE + alt_km
    droe = np.atleast_2d(np.asarray(droe, dtype=float))
    u = np.linspace(0.0, 2.0 * np.pi, n)
    cu, su = np.cos(u), np.sin(u)
    dR = -a_c * (np.outer(droe[:, 2], cu) + np.outer(droe[:, 3], su))
    dT = a_c * (droe[:, 1][:, None] + 2.0 * np.outer(droe[:, 2], su)
                - 2.0 * np.outer(droe[:, 3], cu))
    dN = a_c * (np.outer(droe[:, 4], su) - np.outer(droe[:, 5], cu))
    return dR, dT, dN


def link_geometry_stats(droe, n=240, alt_km=ALT_KM):
    """Geometry and inter-satellite-link statistics over one orbit.

    Returns a dict with the formation footprint (extent in R/T/N), the closest
    and farthest pairwise ranges, the median nearest-neighbour distance (nominal
    ISL hop length), and the peak pairwise range rate (ISL Doppler / antenna
    tracking driver). Distances in km, range rate in m/s.
    """
    dR, dT, dN = relative_track(droe, n, alt_km)
    N = dR.shape[0]
    pos = np.stack([dT, dN, dR], axis=2)                 # (N, n, 3)
    a_c = RE + alt_km
    period = 2.0 * np.pi / np.sqrt(MU / a_c**3)
    dt = period / dR.shape[1]                            # du/dt with du = 2pi/n

    min_range, max_range, max_rate = np.inf, 0.0, 0.0
    nn = np.full(N, np.inf)                              # per-sat nearest neighbour (min over orbit)
    for i in range(N):
        for j in range(i + 1, N):
            rng = np.sqrt(((pos[i] - pos[j])**2).sum(axis=1))
            mr = rng.min()
            min_range = min(min_range, mr)
            max_range = max(max_range, rng.max())
            nn[i] = min(nn[i], mr); nn[j] = min(nn[j], mr)
            max_rate = max(max_rate, np.abs(np.diff(rng)).max() / dt)
    return dict(ext_R=dR.max() - dR.min(), ext_T=dT.max() - dT.min(),
                ext_N=dN.max() - dN.min(), min_range=min_range, max_range=max_range,
                nn_median=float(np.median(nn)), max_rate_ms=max_rate * 1e3)


def connectivity(droe, r_link_km, n=240, alt_km=ALT_KM):
    """Persistent-ISL connectivity graph for a given link range.

    Two satellites share an edge only if their range stays <= ``r_link_km`` for
    the WHOLE orbit (a reliable, always-on link). Returns node-degree stats
    (how many neighbours each satellite can talk to), whether the network is
    connected, its hop diameter, and the component count.
    """
    dR, dT, dN = relative_track(droe, n, alt_km)
    N = dR.shape[0]
    pos = np.stack([dT, dN, dR], axis=2)
    adj = [[] for _ in range(N)]
    deg = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if np.sqrt(((pos[i] - pos[j])**2).sum(axis=1)).max() <= r_link_km:
                adj[i].append(j); adj[j].append(i)
                deg[i] += 1; deg[j] += 1

    diameter = 0
    for s in range(N):                       # diameter: BFS from every node
        dist = [-1] * N; dist[s] = 0; queue = [s]; head = 0
        while head < len(queue):
            x = queue[head]; head += 1
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1; queue.append(y)
        diameter = max(diameter, max(dist))   # -1 stays if unreachable
    # component count
    seen = [False] * N
    comps = 0
    for s in range(N):
        if seen[s]:
            continue
        comps += 1; stack = [s]; seen[s] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True; stack.append(y)
    return dict(deg_min=int(deg.min()), deg_med=float(np.median(deg)),
                deg_max=int(deg.max()), connected=(comps == 1),
                n_components=comps, diameter=(diameter if comps == 1 else None))


def nodal_precession(a, e, inc):
    """Secular J2 nodal precession rate dOmega/dt [rad/s] (vectorized)."""
    n = np.sqrt(MU / a**3)
    p = a * (1.0 - e**2)
    return -1.5 * n * J2 * (RE / p)**2 * np.cos(inc)


def sso_drift(oes, chief):
    """Per-satellite nodal precession and drift relative to the chief.

    Returns (omdot, rel) where omdot[i] is dOmega/dt of satellite i and rel[i]
    is omdot[i] - omdot[chief]. A nonzero rel (driven by an inclination offset
    delta-i_x, or a semimajor-axis offset) means the satellite's node walks away
    from the chief's sun-synchronous node, shearing the formation over time.
    """
    omdot = nodal_precession(oes[:, A], oes[:, ECC], oes[:, INC])
    return omdot, omdot - omdot[chief]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def analyze_formation(R0, V0, meta=None, chief=None, eps=0.050, label="",
                      per_deputy=True, report_sso=False):
    """Print the per-deputy and worst-case-pair separation report.

    ``chief`` defaults to the central (zero-amplitude) satellite when ``meta``
    is supplied. ``eps`` is the minimum safe separation in km. Returns the
    (oes, d_oes) tables for further use.
    """
    oes = compute_elements(R0, V0)
    if chief is None:
        chief = center_index(meta) if meta is not None else 0
    a_c = oes[chief, A]
    d_oes = relative_roe(oes, chief)
    N = R0.shape[0]

    title = f"  [{label}]" if label else ""
    print(f"Formation: N = {N}{title}")
    print(f"Center chief = index {chief};  a_c = {a_c:.3f} km;  eps = {eps * 1e3:.0f} m")

    if per_deputy:
        print("\nPer-deputy separation metrics vs chief (distances in km):")
        hdr = (f"{'idx':>3} {'a_c|de|':>9} {'a_c|di|':>9} {'a_c|dlam|':>10} "
               f"{'sepRN(9)':>9} {'sepRT(13)':>10} {'f(14)':>8} {'mrg(14)':>8}  flags")
        print(hdr)
        print("-" * len(hdr))
        for i in range(N):
            if i == chief:
                continue
            m = separation_metrics(d_oes[i], a_c, eps)
            acde = a_c * np.hypot(d_oes[i][2], d_oes[i][3])
            acdi = a_c * np.hypot(d_oes[i][4], d_oes[i][5])
            flags = "".join([" 9" if m["ok9"] else " .",
                             "13" if m["ok13"] else " .",
                             "14" if m["ok14"] else " ."])
            print(f"{i:>3} {acde:>9.4f} {acdi:>9.4f} {m['aclam']:>10.4f} "
                  f"{m['sep_rn']:>9.4f} {m['sep_rt']:>10.4f} {m['f14']:>8.4f} "
                  f"{m['margin14']:>8.4f}  {flags}")

    # Worst case over all pairs. ROE between two deputies is the difference of
    # their ROE w.r.t. the chief (Eq. 15 -- exact here since all share a_c, i_c).
    min_rn, min_rt, min_margin14 = np.inf, np.inf, np.inf
    fails9 = fails13 = fails14 = npairs = 0
    for j in range(N):
        for k in range(j + 1, N):
            m = separation_metrics(d_oes[k] - d_oes[j], a_c, eps)
            npairs += 1
            min_rn = min(min_rn, m["sep_rn"])
            min_rt = min(min_rt, m["sep_rt"])
            min_margin14 = min(min_margin14, m["margin14"])
            fails9 += not m["ok9"]
            fails13 += not m["ok13"]
            fails14 += not m["ok14"]
    print(f"\nWorst case over all {npairs} spacecraft pairs (eps = {eps * 1e3:.0f} m):")
    print(f"  Eq. (9)  min RN-plane separation : {min_rn:10.4f} km   "
          f"({fails9}/{npairs} pairs below eps)")
    print(f"  Eq. (13) min RT-plane separation : {min_rt:10.4f} km   "
          f"({fails13}/{npairs} pairs below eps)")
    print(f"  Eq. (14) min along-track margin  : {min_margin14:10.4f} km   "
          f"({fails14}/{npairs} pairs violating bound)")

    if report_sso:
        omdot, rel = sso_drift(oes, chief)
        deg_day = np.degrees(omdot[chief]) * 86400.0
        sso_target = np.degrees(SSO_RATE) * 86400.0
        worst = int(np.argmax(np.abs(rel)))
        rel_deg_day = np.degrees(rel[worst]) * 86400.0
        shear_km_yr = abs(rel[worst]) * 365.25 * 86400.0 * a_c   # a_c * d(dOmega)/dt over 1 yr
        ltan_min_day = rel[worst] / (2 * np.pi) * 86400.0 * 1440.0  # node->local-time drift
        print(f"\nJ2 sun-synchronicity (secular nodal precession):")
        print(f"  chief dOmega/dt        : {deg_day:+8.4f} deg/day   "
              f"(SSO target {sso_target:+.4f} deg/day)")
        print(f"  worst rel nodal drift  : {rel_deg_day * 1e3:+8.4f} mdeg/day   (sat #{worst})")
        print(f"  -> formation node shear: {shear_km_yr:8.2f} km/yr   "
              f"LTAN drift {ltan_min_day:+.3f} min/day")
    return oes, d_oes
