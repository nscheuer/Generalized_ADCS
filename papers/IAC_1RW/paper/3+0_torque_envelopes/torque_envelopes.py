#!/usr/bin/env python3
"""Create compact 3-MTQ dipole and torque-envelope illustrations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
M_MAX = 0.20
B_NOW = np.array([22e-6, -8e-6, 41e-6])
B_START = np.array([34e-6, 18e-6, -22e-6])
TORQUE_LIMIT = 1.5e-5
FIGSIZE = (2.45, 2.10)


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


def torque_plane(b_field):
    """Return the exact bounded torque polygon from tau = m x B."""
    dipoles = np.array(np.meshgrid(*[[-M_MAX, M_MAX]] * 3)).T.reshape(-1, 3)
    torques = np.cross(dipoles, b_field)
    center = torques.mean(axis=0)
    u, v = plane_basis(b_field)
    projected = np.column_stack(((torques - center) @ u, (torques - center) @ v))
    hull = convex_hull_2d(projected)
    return center + hull[:, 0, None] * u + hull[:, 1, None] * v


def cube_faces(half_width):
    vertices = np.array(np.meshgrid(*[[-half_width, half_width]] * 3)).T.reshape(-1, 3)
    indices = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
               (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return vertices, [[vertices[i] for i in face] for face in indices]


def setup(title, labels=True):
    figure = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set(xlim=(-TORQUE_LIMIT, TORQUE_LIMIT), ylim=(-TORQUE_LIMIT, TORQUE_LIMIT),
           zlim=(-TORQUE_LIMIT, TORQUE_LIMIT))
    ax.set_box_aspect((1, 1, 1))
    if labels:
        for setter, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                                  (r"$x$", r"$y$", r"$z$")):
            setter(label, labelpad=-5, fontsize=5.5)
    ax.set_title(title, pad=0, fontsize=7.5)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", labelsize=5.5, pad=-5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.get_offset_text().set_fontsize(5.5)
    return figure, ax


def save(figure, stem):
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def add_plane(ax, b_field, color="#2878B5", alpha=0.22, linewidth=1.0):
    polygon = torque_plane(b_field)
    ax.add_collection3d(Poly3DCollection([polygon], facecolor=to_rgba(color, alpha),
                                          edgecolor=color, linewidth=linewidth))
    closed = np.vstack((polygon, polygon[0]))
    ax.plot(*closed.T, color=color, linewidth=linewidth)
    return polygon


def add_vector(ax, vector, color, length=1.1e-5, label=None):
    direction = unit(vector) * length
    return ax.quiver(0, 0, 0, *direction, color=color, linewidth=1.4,
                     arrow_length_ratio=0.12, label=label)


def plot_m_cube():
    figure, ax = setup(r"3 MTQ: produced $m$", labels=False)
    _, faces = cube_faces(0.82e-5)
    ax.add_collection3d(Poly3DCollection(faces, facecolor=to_rgba("#7E3F98", 0.24),
                                         edgecolor="#5B2C6F", linewidth=0.9))
    ax.legend([Patch(facecolor=to_rgba("#7E3F98", 0.24), edgecolor="#5B2C6F")],
              [r"$m_x,m_y,m_z\in[-m_{max},m_{max}]"], loc="upper left",
              bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5)
    save(figure, "01_m_cube")


def plot_m_b_plane():
    figure, ax = setup(r"$m$, $B$, and $m\times B$")
    _, faces = cube_faces(0.82e-5)
    ax.add_collection3d(Poly3DCollection(faces, facecolor=to_rgba("#7E3F98", 0.12),
                                         edgecolor="#7E3F98", linewidth=0.6))
    add_plane(ax, B_NOW)
    add_vector(ax, B_NOW, "#C0392B", label=r"$B$ direction")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5,
              handlelength=0.9, labelspacing=0.12)
    save(figure, "02_m_b_torque_plane")


def plot_b_plane():
    figure, ax = setup("3 MTQ: torque plane")
    add_plane(ax, B_NOW)
    add_vector(ax, B_NOW, "#C0392B", label=r"$B$ direction")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False, fontsize=5.5,
              handlelength=0.9)
    save(figure, "03_b_torque_plane")


def plot_plane_only():
    figure, ax = setup("3 MTQ: resultant torque")
    add_plane(ax, B_NOW)
    save(figure, "04_torque_plane")


def plot_progression(include_vectors):
    figure, ax = setup("3 MTQ: torque-plane progression")
    fields = np.linspace(B_START, B_NOW, 5)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(fields)))
    handles = []
    for index, (field, color) in enumerate(zip(fields, colors), start=1):
        add_plane(ax, field, color=color, alpha=0.12 + 0.04 * index, linewidth=0.65)
        if include_vectors:
            add_vector(ax, field, color, label=f"B{index}")
        handles.append(Patch(facecolor=to_rgba(color, 0.22), edgecolor=color))
    labels = ["B1 old", "B2", "B3", "B4", "B5 now"]
    if include_vectors:
        handles = [Line2D([0], [0], color=color, linewidth=1.4) for color in colors]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False,
              fontsize=5.5, handlelength=0.9, handletextpad=0.35, labelspacing=0.12,
              ncol=2)
    save(figure, "06_plane_progression_with_B" if include_vectors else "05_plane_progression")


def main():
    plot_m_cube()
    plot_m_b_plane()
    plot_b_plane()
    plot_plane_only()
    plot_progression(False)
    plot_progression(True)


if __name__ == "__main__":
    main()
