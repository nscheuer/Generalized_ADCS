#!/usr/bin/env python3
"""Create a 3D torque authority visualization for a 3 MTQ + 1 RW satellite.

The figure shows:
  * a five-frame trail of the bounded 3-axis MTQ torque envelope as B moves;
  * the single reaction-wheel torque line along the body x-axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path(__file__).resolve().parent
BODY_FONT_SIZE = 5.5
TITLE_FONT_SIZE = 7.5
AXIS_LIMIT = 1.5e-5


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return vector / norm


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane normal to normal."""
    normal = unit(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    u = unit(np.cross(normal, reference))
    v = unit(np.cross(normal, u))
    return u, v


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew monotonic-chain hull for small 2D point sets."""
    pts = sorted(set(map(tuple, np.round(points, 14))))
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


def mtq_torque_corners(b_vec: np.ndarray, m_max: float) -> np.ndarray:
    dipole_corners = np.array(
        np.meshgrid([-m_max, m_max], [-m_max, m_max], [-m_max, m_max])
    ).T.reshape(-1, 3)
    return np.cross(dipole_corners, b_vec)


def mtq_envelope_polygon(b_vec: np.ndarray, m_max: float) -> np.ndarray:
    corners = mtq_torque_corners(b_vec, m_max)
    center = corners.mean(axis=0)
    u, v = plane_basis(b_vec)
    points_2d = np.column_stack(((corners - center) @ u, (corners - center) @ v))
    hull_2d = convex_hull_2d(points_2d)
    return center + hull_2d[:, 0, None] * u + hull_2d[:, 1, None] * v


def set_equal_limits(ax, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))


def make_plot(
    b_vec: np.ndarray,
    b_start: np.ndarray,
    m_max: float,
    rw_torque_max: float,
    output_stem: str,
    dpi: int,
    figsize: tuple[float, float],
) -> list[Path]:
    rw_axis = np.array([1.0, 0.0, 0.0])
    b_history = np.linspace(b_start, b_vec, 5)
    mtq_polys = [mtq_envelope_polygon(b, m_max) for b in b_history]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    alphas = np.linspace(0.018, 0.16, len(mtq_polys))
    edge_alphas = np.linspace(0.07, 0.38, len(mtq_polys))
    linewidths = np.linspace(0.35, 0.95, len(mtq_polys))
    for mtq_poly, alpha, edge_alpha, linewidth in zip(mtq_polys, alphas, edge_alphas, linewidths):
        envelope = Poly3DCollection(
            [mtq_poly],
            facecolor=to_rgba("#2E86AB", alpha),
            edgecolor=to_rgba("#155E75", edge_alpha),
            linewidth=linewidth,
            zorder=1,
        )
        ax.add_collection3d(envelope)
        ax.plot(
            np.r_[mtq_poly[:, 0], mtq_poly[0, 0]],
            np.r_[mtq_poly[:, 1], mtq_poly[0, 1]],
            np.r_[mtq_poly[:, 2], mtq_poly[0, 2]],
            color=to_rgba("#155E75", edge_alpha),
            linewidth=linewidth,
            zorder=2,
        )

    rw_end = rw_torque_max * rw_axis
    ax.plot(
        [-rw_end[0], rw_end[0]],
        [-rw_end[1], rw_end[1]],
        [-rw_end[2], rw_end[2]],
        color="#C0392B",
        linewidth=2.1,
        label="RW torque line",
        zorder=100,
    )

    set_equal_limits(ax, AXIS_LIMIT)
    ax.set_xlabel(r"$\tau_x$ [N m]", labelpad=-5, fontsize=BODY_FONT_SIZE)
    ax.set_ylabel(r"$\tau_y$ [N m]", labelpad=-5, fontsize=BODY_FONT_SIZE)
    ax.set_zlabel(r"$\tau_z$ [N m]", labelpad=-5, fontsize=BODY_FONT_SIZE)
    ax.set_title("3 MTQ + 1 RW Torque", pad=0, fontsize=TITLE_FONT_SIZE)
    ax.view_init(elev=24, azim=-43)
    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=BODY_FONT_SIZE, pad=-5)
    ax.xaxis.get_offset_text().set_fontsize(BODY_FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(BODY_FONT_SIZE)
    ax.zaxis.get_offset_text().set_fontsize(BODY_FONT_SIZE)

    handles = [
        Patch(
            facecolor=to_rgba("#2E86AB", alpha),
            edgecolor=to_rgba("#155E75", edge_alpha),
        )
        for alpha, edge_alpha in zip(alphas, edge_alphas)
    ]
    handles.append(Line2D([0], [0], color="#C0392B", linewidth=2.1))
    labels = ["B1 old", "B2", "B3", "B4", "B5 now", "RW x"]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.00),
        frameon=False,
        fontsize=BODY_FONT_SIZE,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.12,
        borderaxespad=0.0,
        columnspacing=0.55,
        ncol=2,
    )

    output_paths = [OUT_DIR / f"{output_stem}.png", OUT_DIR / f"{output_stem}.pdf"]
    for path in output_paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def parse_vector(values: list[float], name: str) -> np.ndarray:
    vector = np.array(values, dtype=float)
    if vector.shape != (3,) or np.linalg.norm(vector) <= 0.0:
        raise argparse.ArgumentTypeError(f"{name} must contain three values and be nonzero.")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D torque visualization for a 3 MTQ + 1 RW satellite."
    )
    parser.add_argument(
        "--B",
        nargs=3,
        type=float,
        default=[22e-6, -8e-6, 41e-6],
        metavar=("BX", "BY", "BZ"),
        help="Current magnetic field vector in body coordinates [T].",
    )
    parser.add_argument(
        "--B-start",
        nargs=3,
        type=float,
        default=[-18e-6, 29e-6, 34e-6],
        metavar=("BX", "BY", "BZ"),
        help="Oldest magnetic field vector in the five-frame trail [T].",
    )
    parser.add_argument("--m-max", type=float, default=0.20, help="MTQ dipole limit per axis [A m^2].")
    parser.add_argument("--rw-torque-max", type=float, default=1.2e-5, help="RW torque limit [N m].")
    parser.add_argument("--output-stem", default="torque_visualization_3mtq_1rw")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[2.45, 2.10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    args = parser.parse_args()

    if args.m_max <= 0.0:
        raise ValueError("--m-max must be positive.")
    if args.rw_torque_max <= 0.0:
        raise ValueError("--rw-torque-max must be positive.")

    output_paths = make_plot(
        b_vec=parse_vector(args.B, "--B"),
        b_start=parse_vector(args.B_start, "--B-start"),
        m_max=args.m_max,
        rw_torque_max=args.rw_torque_max,
        output_stem=args.output_stem,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
