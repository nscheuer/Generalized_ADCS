#!/usr/bin/env python3
"""Create a 3x3 comparison of 3+0, 3+1, and 3+3 torque authority."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import BLUE, ORANGE, RED, configure_ieee_style

configure_ieee_style()


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MTQ_DIPOLE_MAX = 0.20
RW_TORQUE_MAX = 1.2e-5
B_NOW = np.array([22e-6, -8e-6, 41e-6])
B_NEXT = np.array([12e-6, 31e-6, 16e-6])
AXIS_LIMIT = 2.1e-5


def unit(vector):
    return vector / np.linalg.norm(vector)


def plane_basis(normal):
    normal = unit(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(normal, reference))
    return u, unit(np.cross(normal, u))


def hull_2d(points):
    points = sorted(set(map(tuple, np.round(points, 14))))

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
    projected = np.column_stack(((torques - center) @ u, (torques - center) @ v))
    boundary = hull_2d(projected)
    return center + boundary[:, 0, None] * u + boundary[:, 1, None] * v


def rw_vertices(number_rw):
    if number_rw == 0:
        return np.zeros((1, 3))
    if number_rw == 1:
        return np.array([[-RW_TORQUE_MAX, 0, 0], [RW_TORQUE_MAX, 0, 0]])
    return np.array(np.meshgrid(*[[-RW_TORQUE_MAX, RW_TORQUE_MAX]] * 3)).T.reshape(-1, 3)


def combined_points(b_field, number_rw):
    plane = mtq_plane(b_field)
    return (plane[:, None, :] + rw_vertices(number_rw)[None, :, :]).reshape(-1, 3)


def add_polygon(ax, polygon, color=BLUE, alpha=0.24):
    ax.add_collection3d(Poly3DCollection([polygon], facecolor=to_rgba(color, alpha),
                                          edgecolor=color, linewidth=0.65))
    closed = np.vstack((polygon, polygon[0]))
    ax.plot(*closed.T, color=color, linewidth=0.65)


def add_rw(ax, number_rw):
    if number_rw == 1:
        line = rw_vertices(1)
        ax.plot(*line.T, color=RED, linewidth=1.2)
    elif number_rw == 3:
        vertices = rw_vertices(3)
        faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
        ax.add_collection3d(Poly3DCollection([[vertices[i] for i in face] for face in faces],
                                              facecolor=to_rgba(ORANGE, 0.16),
                                              edgecolor=ORANGE, linewidth=0.4))


def add_combined(ax, b_field, number_rw, color=BLUE, alpha=0.20):
    if number_rw == 0:
        add_polygon(ax, mtq_plane(b_field), color, alpha)
        return
    points = combined_points(b_field, number_rw)
    hull = ConvexHull(points)
    triangles = [[points[index] for index in simplex] for simplex in hull.simplices]
    ax.add_collection3d(Poly3DCollection(triangles, facecolor=to_rgba(color, alpha),
                                          edgecolor=color, linewidth=0.28))


def setup(ax, title):
    ax.set(xlim=(-AXIS_LIMIT, AXIS_LIMIT), ylim=(-AXIS_LIMIT, AXIS_LIMIT),
           zlim=(-AXIS_LIMIT, AXIS_LIMIT))
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, pad=0, fontsize=7)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.10)
    ax.tick_params(axis="both", labelsize=4.5, pad=-4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.get_offset_text().set_fontsize(4.5)


def make_plot():
    configurations = (("3+0", 0), ("3+1", 1), ("3+3", 3))
    figure = plt.figure(figsize=(7.0, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("white")

    for column, (name, number_rw) in enumerate(configurations):
        # Row 1: the individual actuator authorities.
        ax = figure.add_subplot(3, 3, column + 1, projection="3d")
        setup(ax, name)
        add_polygon(ax, mtq_plane(B_NOW))
        add_rw(ax, number_rw)

        # Rows 2 and 3: the combined authority at two field states.
        for row, b_field in ((1, B_NOW), (2, B_NEXT)):
            ax = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            setup(ax, name)
            add_combined(ax, b_field, number_rw)

    figure.text(0.01, 0.67, "Separate", rotation=90, va="center", fontsize=8)
    figure.text(0.01, 0.39, "Instant 1", rotation=90, va="center", fontsize=8)
    figure.text(0.01, 0.12, "Instant 2", rotation=90, va="center", fontsize=8)
    figure.text(0.5, 0.01, r"Torque coordinates [N m]", ha="center", fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"torque_envelopes_3x3.{extension}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    make_plot()
