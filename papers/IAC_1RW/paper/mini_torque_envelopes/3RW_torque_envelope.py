#!/usr/bin/env python3
"""Generate the compact three-reaction-wheel torque-envelope figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_plot():
    torque_limit = 1.2e-5
    vertices = [(x * torque_limit, y * torque_limit, z * torque_limit)
                for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    figure = plt.figure(figsize=(2.45, 2.10), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    figure.patch.set_facecolor("white")
    ax.add_collection3d(Poly3DCollection([[vertices[i] for i in face] for face in faces],
                                          facecolor="#D97706", alpha=0.20,
                                          edgecolor="#92400E", linewidth=0.8))
    limit = 1.5e-5
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    for axis, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                           (r"$\tau_x$ [N m]", r"$\tau_y$ [N m]", r"$\tau_z$ [N m]")):
        axis(label, labelpad=-5, fontsize=5.5)
    ax.set_title("3 RW: available torque", pad=0, fontsize=7.5)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", labelsize=5.5, pad=-5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.get_offset_text().set_fontsize(5.5)
    ax.legend([Line2D([0], [0], color="#D97706", linewidth=4)], ["RW torque envelope"],
              loc="upper left", bbox_to_anchor=(-0.01, 1), frameon=False,
              fontsize=5.5, handlelength=0.9)
    for extension in ("png", "pdf"):
        figure.savefig(OUT_DIR / f"3RW_torque_envelope.{extension}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    make_plot()
