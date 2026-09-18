#!/usr/bin/env python3
"""Plot the torque authority of a 3-MTQ + 1-RW actuator cluster."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import BLUE, ORANGE, RED, configure_ieee_style

configure_ieee_style()


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RW_TORQUE_MAX = 1.2e-5
MTQ_DIPOLE_MAX = 0.20 / 2.0
B_NOW = np.array([22e-6, -8e-6, 41e-6])
B_START = np.array([34e-6, 18e-6, -22e-6])
AXIS_LIMIT = 1.9e-5


def unit(vector):
    return vector / np.linalg.norm(vector)


def plane_basis(normal):
    normal = unit(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(normal, reference))
    return u, unit(np.cross(normal, u))


def convex_hull_2d(points):
    points = sorted(set(map(tuple, np.round(points, 14))))
    if len(points) <= 1:
        return np.array(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def mtq_plane(b_field):
    dipoles = np.array(np.meshgrid(*[[-MTQ_DIPOLE_MAX, MTQ_DIPOLE_MAX]] * 3)).T.reshape(-1, 3)
    torques = np.cross(dipoles, b_field)
    center = torques.mean(axis=0)
    u, v = plane_basis(b_field)
    points_2d = np.column_stack(((torques - center) @ u, (torques - center) @ v))
    hull = convex_hull_2d(points_2d)
    return center + hull[:, 0, None] * u + hull[:, 1, None] * v


def rw_line():
    return np.array([[-RW_TORQUE_MAX, 0.0, 0.0], [RW_TORQUE_MAX, 0.0, 0.0]])


def minkowski_points(b_field):
    """Return all vertex sums of the MTQ polygon and RW line segment."""
    return (mtq_plane(b_field)[:, None, :] + rw_line()[None, :, :]).reshape(-1, 3)


def setup(title):
    figure = plt.figure(figsize=(2.55, 2.15), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set(xlim=(-AXIS_LIMIT, AXIS_LIMIT), ylim=(-AXIS_LIMIT, AXIS_LIMIT),
           zlim=(-AXIS_LIMIT, AXIS_LIMIT))
    ax.set_box_aspect((1, 1, 1))
    for setter, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                             (r"$\tau_x$ [N m]", r"$\tau_y$ [N m]", r"$\tau_z$ [N m]")):
        setter(label, labelpad=-5, fontsize=6.5)
    ax.set_title(title, pad=0, fontsize=8.0)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", labelsize=6.0, pad=-5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.get_offset_text().set_fontsize(6.0)
    return figure, ax


def add_plane(ax, b_field, color=BLUE, alpha=0.24):
    plane = mtq_plane(b_field)
    ax.add_collection3d(Poly3DCollection([plane], facecolor=to_rgba(color, alpha),
                                          edgecolor=color, linewidth=0.9))
    closed = np.vstack((plane, plane[0]))
    ax.plot(*closed.T, color=color, linewidth=0.9)


def add_rw_line(ax, color=RED, linewidth=1.6):
    line = rw_line()
    ax.plot(*line.T, color=color, linewidth=linewidth)


def add_volume(ax, b_field, color=BLUE, alpha=0.22):
    points = minkowski_points(b_field)
    hull = ConvexHull(points)
    triangles = [[points[index] for index in simplex] for simplex in hull.simplices]
    ax.add_collection3d(Poly3DCollection(triangles, facecolor=to_rgba(color, alpha),
                                         edgecolor=color, linewidth=0.45))


def save(figure, stem):
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_separate():
    figure, ax = setup("3 MTQ + 1 RW: instant authority")
    add_plane(ax, B_NOW)
    add_rw_line(ax)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.24), edgecolor=BLUE),
               Line2D([0], [0], color=RED, linewidth=1.6)],
              ["MTQ torque plane", "RW torque line"], loc="upper left",
              bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5,
              handlelength=0.9, labelspacing=0.12)
    save(figure, "01_mtq_plane_rw_separate")


def plot_combined():
    figure, ax = setup("3 MTQ + 1 RW: combined authority")
    add_volume(ax, B_NOW)
    ax.legend([Patch(facecolor=to_rgba(BLUE, 0.22), edgecolor=BLUE)],
              ["MTQ plane $\u2295$ RW line"], loc="upper left", bbox_to_anchor=(-0.01, 1),
              frameon=False, fontsize=5.5, handlelength=0.9)
    save(figure, "02_mtq_rw_minkowski_sum")


def plot_plane_progression():
    figure, ax = setup("3 MTQ + 1 RW: plane progression")
    fields = np.linspace(B_START, B_NOW, 5)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(fields)))
    for field, color in zip(fields, colors):
        add_plane(ax, field, color=color, alpha=0.12)
    add_rw_line(ax)
    handles = [Patch(facecolor=to_rgba(color, 0.18), edgecolor=color) for color in colors]
    handles.append(Line2D([0], [0], color=RED, linewidth=1.6))
    ax.legend(handles, ["B1 old", "B2", "B3", "B4", "B5 now", "RW"], loc="upper left",
              bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5, handlelength=0.9,
              handletextpad=0.35, labelspacing=0.12, ncol=2)
    save(figure, "03_mtq_plane_progression_rw")


def plot_combined_progression():
    figure, ax = setup("3 MTQ + 1 RW: combined progression")
    fields = np.linspace(B_START, B_NOW, 5)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(fields)))
    for field, color in zip(fields, colors):
        add_volume(ax, field, color=color, alpha=0.13)
    handles = [Line2D([0], [0], color=color, linewidth=2.0) for color in colors]
    ax.legend(handles, ["B1 old", "B2", "B3", "B4", "B5 now"], loc="upper left",
              bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5, handlelength=0.9,
              handletextpad=0.35, labelspacing=0.12, ncol=2)
    save(figure, "04_combined_progression")


def main():
    plot_separate()
    plot_combined()
    plot_plane_progression()
    plot_combined_progression()


if __name__ == "__main__":
    main()
