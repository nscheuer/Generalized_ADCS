"""Paper 1 -- FIG-ALLOC + TAB-ALLOC + DATA-ALLOC-TIMING (Sections V-E, VI-A).

Allocation-method comparison: LP vs QP vs constrained-QP. For many random
desired-torque directions/magnitudes and magnetic-field geometries, drive
each controller's ``allocate_max_torque_in_direction`` primitive directly
(it is self-contained -- no closed-loop sim needed), then measure:

  * direction error  -- angle between achieved and desired torque
                         (LP preserves it ~0; QP can exceed 30 deg)
  * magnitude ratio   -- delivered torque along the desired direction /
                         requested magnitude
  * solve time [us]   -- the [TODO-DATA] in Section VI-A

Achieved torque is reconstructed with the framework's own torque map
(paper Eq. 9): tau = A_RW u_rw - [B]x A_MTQ u_mtq. Nothing in the framework
is modified (this is a non-invasive analysis harness).

Emits:
  * output_data/fig_alloc.png            -> FIG-ALLOC (dir-err + mag-ratio)
  * output_data/tab_alloc.{tex,csv,md}   -> TAB-ALLOC + the VI-A us numbers

Single knob: ``PAPER1_SCALE=fast`` (200 samples) | ``paper`` (5000).
"""

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP, MTQ_w_RW_QP
from ADCS.controller.mtq_w_rw_QPC import MTQ_w_RW_QPC
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers import metrics as M

from papers.Generalized_ACS._paper1_sim import build_actuators


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors (self-contained; no dependency on the
    framework ``normalize`` signature)."""
    return v / np.linalg.norm(v, axis=1, keepdims=True)

OUTPUT_DIR = "papers/Generalized_ACS/output_data"
CONFIG = "3MTQ+1RW"
N_SAMPLES = {"fast": 200, "paper": 5000}
B_MAG = 3.0e-5            # representative LEO field magnitude [T]
TAU_REF = 0.4 * B_MAG     # ~ mtq_max * |B|: order of achievable torque [N m]


def _satellite() -> Satellite:
    acts, _ = build_actuators(CONFIG)
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=1.2, J_0=np.diagflat([0.022, 0.022, 0.004]),
                     actuators=acts, sensors=mtms,
                     boresight=np.array([0, 0, 1]))


def _torque_map(sat: Satellite):
    """A_RW (3,n_rw) and A_MTQ axes (3,n_mtq) in the controller's actuator
    order (RWs then MTQs, filtered exactly as the allocator does)."""
    rws = [a for a in sat.actuators if isinstance(a, RW)]
    mtqs = [a for a in sat.actuators if isinstance(a, MTQ)]
    A_rw = np.array([np.asarray(a.axis, float) for a in rws]).T.reshape(3, -1) \
        if rws else np.zeros((3, 0))
    A_mtq = np.array([np.asarray(a.axis, float) for a in mtqs]).T.reshape(3, -1) \
        if mtqs else np.zeros((3, 0))
    return A_rw, A_mtq


def _achieved(A_rw, A_mtq, u_rw, u_mtq, b_body) -> np.ndarray:
    """Framework Eq. 9 torque: A_RW u_rw + sum_j (-B x axis_j) u_mtq_j."""
    tau = A_rw @ np.asarray(u_rw, float) if A_rw.shape[1] else np.zeros(3)
    if A_mtq.shape[1]:
        # MTQ torque per actuator = -cross(b_body, axis) * u  (see
        # magnetotorquer.py); vectorized over the MTQ axes:
        tau = tau + (-np.cross(b_body, A_mtq.T) * np.asarray(u_mtq, float)[:, None]).sum(0)
    return tau


def main() -> None:
    scale = os.environ.get("PAPER1_SCALE", "fast")
    n = N_SAMPLES[scale]
    print(f"[ALLOC] config={CONFIG} samples={n}")

    sat = _satellite()
    A_rw, A_mtq = _torque_map(sat)
    methods = {
        "LP": MTQ_w_RW_LP(est_sat=sat, p_gain=5e-5, d_gain=1e-3,
                          c_gain=1e-3, h_target=np.zeros(3)),
        "QP": MTQ_w_RW_QP(est_sat=sat, p_gain=5e-5, d_gain=1e-3,
                          c_gain=1e-3, h_target=np.zeros(3)),
        "cQP": MTQ_w_RW_QPC(est_sat=sat, p_gain=5e-5, d_gain=1e-3,
                            c_gain=1e-3, h_target=np.zeros(3)),
    }

    rng = np.random.default_rng(0)
    # Magnitudes straddle the achievable envelope so the LP/QP direction
    # behaviour near the boundary (the paper's key result) is exercised.
    tau_dirs = _unit_rows(rng.standard_normal((n, 3)))
    tau_mags = TAU_REF * rng.uniform(0.2, 2.0, size=n)
    tau_des = tau_dirs * tau_mags[:, None]
    b_dirs = _unit_rows(rng.standard_normal((n, 3))) * B_MAG

    rows: List[Dict[str, Any]] = []
    dir_err: Dict[str, np.ndarray] = {}
    mag_ratio: Dict[str, np.ndarray] = {}
    for name, ctrl in methods.items():
        de = np.full(n, np.nan)
        mr = np.full(n, np.nan)
        t0 = time.perf_counter()
        for i in range(n):
            td, bb = tau_des[i], b_dirs[i]
            if name == "cQP":
                u_rw, u_mtq, _ = ctrl.allocate_max_torque_in_direction(
                    td, bb, sat, np.zeros(3), np.zeros(max(1, A_rw.shape[1])))
            else:
                u_rw, u_mtq, _ = ctrl.allocate_max_torque_in_direction(
                    td, bb, sat)
            tau_ach = _achieved(A_rw, A_mtq, u_rw, u_mtq, bb)
            de[i] = M.allocation_direction_error_deg(tau_ach, td)[0]
            mr[i] = M.allocation_magnitude_ratio(tau_ach, td)[0]
        us = 1e6 * (time.perf_counter() - t0) / n

        dir_err[name], mag_ratio[name] = de, mr
        rows.append({
            "method": name,
            "mean_dir_err_deg": float(np.nanmean(de)),
            "median_dir_err_deg": float(np.nanmedian(de)),
            "p95_dir_err_deg": float(np.nanpercentile(de, 95)),
            "mean_mag_ratio": float(np.nanmean(mr)),
            "solve_time_us": float(us),
        })
        print(f"  {name}: dir_err mean {rows[-1]['mean_dir_err_deg']:.3f} deg, "
              f"mag_ratio {rows[-1]['mean_mag_ratio']:.3f}, "
              f"{us:.1f} us/solve")

    cols = ["method", "mean_dir_err_deg", "median_dir_err_deg",
            "p95_dir_err_deg", "mean_mag_ratio", "solve_time_us"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[TAB-ALLOC] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_alloc"),
                         columns=cols), sep="\n  ")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(methods)
    a1.boxplot([dir_err[m][np.isfinite(dir_err[m])] for m in names],
               tick_labels=names, showmeans=True)
    a1.set_ylabel("Direction error [deg]")
    a1.set_title("FIG-ALLOC: torque direction error")
    a1.grid(True, ls="--", alpha=0.5)
    a2.boxplot([mag_ratio[m][np.isfinite(mag_ratio[m])] for m in names],
               tick_labels=names, showmeans=True)
    a2.set_ylabel("Magnitude ratio")
    a2.set_title("FIG-ALLOC: delivered/desired magnitude")
    a2.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_alloc.png")
    fig.savefig(p, dpi=150)
    print("[FIG-ALLOC] wrote", p)


if __name__ == "__main__":
    main()
