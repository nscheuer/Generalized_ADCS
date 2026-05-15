"""Paper 1 -- FIG-POLYTOPE (Section IV-A).

Static export of the torque polytopes for MTQ-only, 3MTQ+1RW and 3MTQ+3RW,
shown at a single instant and accumulated over one orbit. The existing
``generate_reachable_sets.py`` is an interactive slider explorer; this
reuses its ``ActuatorModel`` / ``GeometryUtils`` (no duplication) to emit a
publication figure non-interactively.

The orbital field variation is represented by sweeping the body-frame
magnetic-field direction over the sphere (the geomagnetic vector rotates
through ~all directions over an orbit). The point the figure makes: the
MTQ-only instantaneous set is a thin 2-D disk perpendicular to B; over an
orbit it sweeps into a 3-D volume -- i.e. magnetorquer-only attitude
control becomes fully 3-axis given orbital motion.

Emits:
  * output_data/fig_polytope.png
  * output_data/tab_polytope.{tex,csv,md}  (accumulated hull volume / config)

Single knob: ``PAPER1_SCALE=fast`` (60 field dirs) | ``paper`` (400).
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np
from scipy.spatial import ConvexHull

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers import metrics as M
from papers.Generalized_ACS.generate_reachable_sets import (
    ActuatorModel, GeometryUtils,
)

OUTPUT_DIR = "papers/Generalized_ACS/output_data"
M_MAX = 0.4            # MTQ max dipole [A m^2]
RW_UMAX = 7e-3        # RW max torque [N m]
B_MAG = 3.0e-5        # representative |B| [T]
N_DIRS = {"fast": 60, "paper": 400}

# axes for the RW sets per config (body frame)
RW_AXES = {
    "MTQ-only": [],
    "3MTQ+1RW": [np.array([0.0, 0.0, 1.0])],
    "3MTQ+3RW": [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0, 0, 1.0])],
}
CONFIGS = list(RW_AXES)


def _fib_sphere(n: int) -> np.ndarray:
    """Deterministic ~uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(th),
                            np.sin(phi) * np.sin(th), np.cos(phi)])


def _config_points(B_dirs: np.ndarray, rw_axes: List[np.ndarray]):
    """Return (instant_pts, accumulated_pts) torque clouds for a config.

    instant = first field sample; accumulated = union over all samples.
    RW box (time-invariant) is Minkowski-summed onto the MTQ envelope.
    """
    def with_rw(mtq_pts):
        pts = mtq_pts
        for ax in rw_axes:
            pts = GeometryUtils.minkowski_sum(
                pts, ActuatorModel.get_rw_torque_envelope(ax, RW_UMAX))
        return pts

    inst = with_rw(ActuatorModel.get_mtq_torque_envelope(
        B_dirs[0] * B_MAG, M_MAX))
    acc = np.vstack([
        with_rw(ActuatorModel.get_mtq_torque_envelope(d * B_MAG, M_MAX))
        for d in B_dirs
    ])
    return inst, acc


def _hull_volume(pts: np.ndarray) -> float:
    try:
        return float(ConvexHull(pts).volume)
    except Exception:  # degenerate (coplanar) -> 0 volume
        return 0.0


def main() -> None:
    scale = os.environ.get("PAPER1_SCALE", "fast")
    B_dirs = _fib_sphere(N_DIRS[scale])
    print(f"[POLYTOPE] {len(B_dirs)} field directions")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(15, 5))
    rows: List[Dict[str, Any]] = []
    for k, cfg in enumerate(CONFIGS):
        inst, acc = _config_points(B_dirs, RW_AXES[cfg])
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")

        for pts, color, alpha in ((acc, "tab:blue", 0.12),
                                  (inst, "tab:orange", 0.55)):
            if pts.shape[0] >= 4 and _hull_volume(pts) > 0:
                h = ConvexHull(pts)
                ax.add_collection3d(Poly3DCollection(
                    pts[h.simplices], alpha=alpha, facecolor=color,
                    edgecolor="k", linewidths=0.2))
            else:  # degenerate (MTQ-only instant disk)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           s=4, c=color, alpha=0.6)
        lim = np.abs(acc).max() * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_title(cfg)
        ax.set_xlabel("$\\tau_x$")
        ax.set_ylabel("$\\tau_y$")
        rows.append({
            "config": cfg,
            "instant_hull_vol": _hull_volume(inst),
            "orbit_hull_vol": _hull_volume(acc),
        })
        print(f"  {cfg}: instant_vol={rows[-1]['instant_hull_vol']:.3e} "
              f"orbit_vol={rows[-1]['orbit_hull_vol']:.3e}")

    fig.suptitle("FIG-POLYTOPE: torque sets (orange=instant, "
                 "blue=accumulated over one orbit)")
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = os.path.join(OUTPUT_DIR, "fig_polytope.png")
    fig.savefig(p, dpi=150)
    print("[FIG-POLYTOPE] wrote", p)

    print("[TAB-POLYTOPE] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_polytope"),
                         columns=["config", "instant_hull_vol",
                                  "orbit_hull_vol"],
                         float_fmt="{:.3e}"), sep="\n  ")


if __name__ == "__main__":
    main()
