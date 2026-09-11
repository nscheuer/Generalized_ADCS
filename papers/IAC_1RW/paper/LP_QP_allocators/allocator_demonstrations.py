#!/usr/bin/env python3
"""Create 2-D illustrations of LP and QP torque allocation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import BLUE, ORANGE, configure_ieee_style

configure_ieee_style()

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A representative bounded torque envelope, expressed in normalized torque units.
ENVELOPE = np.array([
    [0.70, 1.20],
    [1.40, -0.50],
    [-0.70, -1.20],
    [-1.40, 0.50],
])
TAU_REF = np.array([1.80, 1.35])
TAU_REF_INSIDE = np.array([0.25, 0.20])


def ray_boundary(origin, direction, polygon):
    """Return the first intersection of origin + t*direction with polygon."""
    direction = np.asarray(direction, dtype=float)
    best_t = np.inf
    best_point = None
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        matrix = np.column_stack((direction, -edge))
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        t, u = np.linalg.solve(matrix, start - origin)
        if t >= 0.0 and 0.0 <= u <= 1.0 and t < best_t:
            best_t = t
            best_point = origin + t * direction
    if best_point is None:
        raise ValueError("Ray does not intersect the torque envelope.")
    return best_point


def closest_point_on_segment(point, start, end):
    edge = end - start
    fraction = np.dot(point - start, edge) / np.dot(edge, edge)
    fraction = np.clip(fraction, 0.0, 1.0)
    return start + fraction * edge


def closest_point_in_polygon(point, polygon):
    """Project point onto the boundary of a convex polygon."""
    candidates = [closest_point_on_segment(point, start, end)
                  for start, end in zip(polygon, np.roll(polygon, -1, axis=0))]
    return min(candidates, key=lambda candidate: np.linalg.norm(candidate - point))


def arrow(ax, endpoint, color, label):
    ax.add_patch(FancyArrowPatch((0.0, 0.0), endpoint, arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.4, color=color,
                                 zorder=5, label=label))


def setup(title, tau_ref=TAU_REF, annotate_ref=True):
    figure, ax = plt.subplots(figsize=(3.35, 2.65), constrained_layout=True)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_patch(Polygon(ENVELOPE, closed=True, facecolor=BLUE, alpha=0.16,
                         edgecolor=BLUE, linewidth=1.15, zorder=1))
    ax.scatter(*tau_ref, color=ORANGE, s=19, zorder=7)
    if annotate_ref:
        ax.annotate(r"$\tau_{\mathrm{ref}}$", tau_ref, xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=ORANGE)
    ax.set_xlim(-1.65, 2.15)
    ax.set_ylim(-1.50, 1.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\tau_x$", fontsize=8)
    ax.set_ylabel(r"$\tau_y$", fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, alpha=0.35)
    ax.axhline(0.0, color="#777777", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="#777777", linewidth=0.45, zorder=0)
    ax.tick_params(labelsize=7)
    return figure, ax


def save(figure, stem):
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_lp():
    tau_lp = ray_boundary(np.zeros(2), TAU_REF, ENVELOPE)
    figure, ax = setup("LP allocator: colinear solution")
    ax.plot([0.0, TAU_REF[0]], [0.0, TAU_REF[1]], color=ORANGE, linestyle=":",
            linewidth=1.5, zorder=2)
    arrow(ax, tau_lp, BLUE, r"$\tau_{\mathrm{LP}}$")
    ax.scatter(*tau_lp, color=BLUE, s=14, zorder=6)
    ax.annotate(r"$\tau_{\mathrm{LP}}$", tau_lp, xytext=(5, -12),
                textcoords="offset points", fontsize=8, color=BLUE)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.16), edgecolor=BLUE),
               Line2D([0], [0], color=BLUE, linewidth=1.4),
               Line2D([0], [0], color=ORANGE, linestyle=":", linewidth=1.5)],
              ["Torque envelope", r"$\tau_{\mathrm{LP}}$", r"$\tau_{\mathrm{ref}}$ direction"],
              loc="lower left", frameon=False, fontsize=7, handlelength=1.2)
    save(figure, "01_lp_colinear")


def plot_qp():
    tau_qp = closest_point_in_polygon(TAU_REF, ENVELOPE)
    figure, ax = setup("QP allocator: nearest solution")
    ax.plot([0.0, TAU_REF[0]], [0.0, TAU_REF[1]], color=ORANGE, linestyle=":",
            linewidth=1.5, zorder=2)
    arrow(ax, tau_qp, BLUE, r"$\tau_{\mathrm{QP}}$")
    ax.scatter(*tau_qp, color=BLUE, s=14, zorder=6)
    ax.annotate(r"$\tau_{\mathrm{QP}}$", tau_qp, xytext=(-31, 5),
                textcoords="offset points", fontsize=8, color=BLUE)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.16), edgecolor=BLUE),
               Line2D([0], [0], color=BLUE, linewidth=1.4),
               Line2D([0], [0], color=ORANGE, linestyle=":", linewidth=1.5)],
              ["Torque envelope", r"$\tau_{\mathrm{QP}}$", r"$\tau_{\mathrm{ref}}$ direction"],
              loc="lower left", frameon=False, fontsize=7, handlelength=1.2)
    save(figure, "02_qp_nearest")


def plot_lp_qp_combined():
    """Overlay the colinear LP and nearest-point QP solutions."""
    tau_lp = ray_boundary(np.zeros(2), TAU_REF, ENVELOPE)
    tau_qp = closest_point_in_polygon(TAU_REF, ENVELOPE)
    figure, ax = setup("LP and QP allocators: solution comparison")
    ax.plot([0.0, TAU_REF[0]], [0.0, TAU_REF[1]], color="#666666", linestyle=":",
            linewidth=1.35, zorder=2)
    arrow(ax, tau_lp, BLUE, r"$\tau_{\mathrm{LP}}$")
    arrow(ax, tau_qp, ORANGE, r"$\tau_{\mathrm{QP}}$")
    ax.scatter(*tau_lp, color=BLUE, s=14, zorder=6)
    ax.scatter(*tau_qp, color=ORANGE, s=14, zorder=6)
    ax.annotate(r"$\tau_{\mathrm{LP}}$", tau_lp, xytext=(5, -12),
                textcoords="offset points", fontsize=8, color=BLUE)
    ax.annotate(r"$\tau_{\mathrm{QP}}$", tau_qp, xytext=(-31, 5),
                textcoords="offset points", fontsize=8, color=ORANGE)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.16), edgecolor=BLUE),
               Line2D([0], [0], color=BLUE, linewidth=1.4),
               Line2D([0], [0], color=ORANGE, linewidth=1.4),
               Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.35)],
              ["Torque envelope", r"$\tau_{\mathrm{LP}}$",
               r"$\tau_{\mathrm{QP}}$", r"$\tau_{\mathrm{ref}}$ direction"],
              loc="lower left", frameon=False, fontsize=7, handlelength=1.2)
    save(figure, "05_lp_qp_combined")


def plot_perfect(allocator, stem):
    """Plot an attainable reference, for which LP and QP are both exact."""
    figure, ax = setup(f"{allocator} allocator: perfect solution", TAU_REF_INSIDE,
                       annotate_ref=False)
    arrow(ax, TAU_REF_INSIDE, BLUE, rf"$\tau_{{\mathrm{{{allocator}}}}}$")
    ax.plot([0.0, TAU_REF_INSIDE[0]], [0.0, TAU_REF_INSIDE[1]], color=ORANGE,
            linestyle=":", linewidth=1.5, zorder=2)
    ax.scatter(*TAU_REF_INSIDE, color=ORANGE, s=14, zorder=6)
    ax.annotate(r"$\tau_{\mathrm{ref}}=\tau_{\mathrm{" + allocator + r"}}$",
                TAU_REF_INSIDE, xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=ORANGE)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.16), edgecolor=BLUE),
               Line2D([0], [0], color=BLUE, linewidth=1.4),
               Line2D([0], [0], color=ORANGE, linestyle=":", linewidth=1.5)],
              ["Torque envelope", rf"$\tau_{{\mathrm{{{allocator}}}}}$",
               r"$\tau_{\mathrm{ref}}$ direction"], loc="lower left", frameon=False,
              fontsize=7, handlelength=1.2)
    save(figure, stem)


def main():
    plot_lp()
    plot_qp()
    plot_perfect("LP", "03_lp_perfect")
    plot_perfect("QP", "04_qp_perfect")
    plot_lp_qp_combined()


if __name__ == "__main__":
    main()
