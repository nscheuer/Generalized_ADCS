#!/usr/bin/env python3
"""Generate the compact 3-MTQ torque-envelope figure."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import BLUE, PURPLE, RED, configure_ieee_style

configure_ieee_style()


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def unit(vector):
    return vector / np.linalg.norm(vector)


def plane_basis(normal):
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(unit(normal), reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(normal, reference))
    return u, unit(np.cross(normal, u))


def convex_hull_2d(points):
    """Return the Andrew monotonic-chain hull for a small point set."""
    points = sorted(set(map(tuple, np.round(points, 14))))
    if len(points) <= 1:
        return np.array(points)

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

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


def envelope(b_field, dipole_limit):
    dipoles = np.array(np.meshgrid(*[[-dipole_limit, dipole_limit]] * 3)).T.reshape(-1, 3)
    torques = np.cross(dipoles, b_field)
    center = torques.mean(axis=0)
    u, v = plane_basis(b_field)
    points = np.column_stack(((torques - center) @ u, (torques - center) @ v))
    hull = convex_hull_2d(points)
    return center + hull[:, 0, None] * u + hull[:, 1, None] * v


def make_plot():
    b_field = np.array([22e-6, -8e-6, 41e-6])
    dipole_limit = 0.20
    torque = envelope(b_field, dipole_limit)
    figure = plt.figure(figsize=(2.45, 2.10), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    figure.patch.set_facecolor("white")
    polygon = Poly3DCollection([torque], facecolor=to_rgba(BLUE, 0.22),
                                edgecolor=BLUE, linewidth=0.8)
    ax.add_collection3d(polygon)
    ax.plot(*np.vstack((torque, torque[0])).T, color=BLUE, linewidth=0.8)
    ax.quiver(0, 0, 0, *b_field / np.linalg.norm(b_field) * 1.1e-5,
              color=RED, linewidth=1.2, arrow_length_ratio=0.12)
    limit = 1.5e-5
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    for axis, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                           (r"$\tau_x$ [N m]", r"$\tau_y$ [N m]", r"$\tau_z$ [N m]")):
        axis(label, labelpad=-5, fontsize=6.5)
    ax.set_title("3 MTQ: available torque", pad=0, fontsize=8.0)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", labelsize=6.0, pad=-5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.get_offset_text().set_fontsize(6.0)
    ax.legend([Line2D([0], [0], color=BLUE, linewidth=3),
               Line2D([0], [0], color=RED, linewidth=1.2)],
              ["MTQ torque envelope", r"$B$ direction"], loc="upper left",
              bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5,
              handlelength=0.9, labelspacing=0.12)
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"3MTQ_torque_envelope.{extension}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    make_plot()
