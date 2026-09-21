#!/usr/bin/env python3
"""Render an easier-to-read 2-D companion to the LP/QP torque animations.

Each frame is projected onto the plane spanned by the current reference torque
and QP solution.  The reference therefore lies on the horizontal axis, LP is
visibly colinear with it, and QP's shortest Euclidean correction is shown as a
single orange connector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull
from PIL import Image

from animate_3mtq_torque_polytope import AXIS, CYAN, VIOLET, gif_frames
from animate_lp_qp_torque_allocation import (
    LP_COLOR,
    QP_COLOR,
    REFERENCE_COLOR,
    actuator_map,
    lp_solution,
    qp_solution,
    reference_torque,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "allocator_tracking_assets"
PIXEL_LIMIT = 4.70e-5


def projection_basis(reference: np.ndarray, qp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal reference and vertical QP-error basis vectors."""
    horizontal = reference / np.linalg.norm(reference)
    residual = qp - np.dot(qp, horizontal) * horizontal
    if np.linalg.norm(residual) < 1e-14:
        residual = np.array((0.0, 0.0, 1.0))
        residual -= np.dot(residual, horizontal) * horizontal
    vertical = residual / np.linalg.norm(residual)
    return horizontal, vertical


def project(points: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    return np.column_stack((points @ horizontal, points @ vertical))


def add_arrow(ax: plt.Axes, endpoint: np.ndarray, color: str, linewidth: float, zorder: int) -> None:
    ax.add_patch(FancyArrowPatch(
        (0.0, 0.0), endpoint, arrowstyle="-|>", mutation_scale=13,
        linewidth=linewidth, color=color, shrinkA=0.0, shrinkB=0.0, zorder=zorder,
    ))


def render_frame(wheels: int, phase: float, pixels: int) -> Image.Image:
    matrix, limits = actuator_map(wheels)
    reference = reference_torque(phase)
    lp = lp_solution(matrix, limits, reference)
    qp = qp_solution(matrix, limits, reference)
    horizontal, vertical = projection_basis(reference, qp)
    authority = np.asarray(list(__import__("itertools").product((-1.0, 1.0), repeat=len(limits)))) * limits
    authority = authority @ matrix.T
    envelope = project(authority, horizontal, vertical)
    reference_2d = project(reference[None, :], horizontal, vertical)[0]
    lp_2d = project(lp[None, :], horizontal, vertical)[0]
    qp_2d = project(qp[None, :], horizontal, vertical)[0]

    figure, ax = plt.subplots(figsize=(5.0, 5.0), dpi=pixels / 5.0)
    figure.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_xlim(-PIXEL_LIMIT, PIXEL_LIMIT)
    ax.set_ylim(-PIXEL_LIMIT, PIXEL_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    hull = ConvexHull(envelope)
    polygon = envelope[hull.vertices]
    ax.add_patch(Polygon(
        polygon, closed=True, facecolor=to_rgba(VIOLET, 0.38),
        edgecolor=to_rgba(CYAN, 0.90), linewidth=1.35, zorder=2,
    ))
    ax.axhline(0.0, color=to_rgba(AXIS, 0.50), linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=to_rgba(AXIS, 0.50), linewidth=0.8, zorder=1)

    # Reference arrow, LP ray, QP endpoint and the Euclidean residual.
    ax.plot([0.0, reference_2d[0]], [0.0, reference_2d[1]], color=to_rgba(REFERENCE_COLOR, 0.45),
            linestyle=(0, (3, 3)), linewidth=1.05, zorder=5)
    add_arrow(ax, reference_2d, REFERENCE_COLOR, 1.55, 6)
    add_arrow(ax, lp_2d, LP_COLOR, 2.8, 8)
    add_arrow(ax, qp_2d, QP_COLOR, 2.8, 9)
    ax.plot([qp_2d[0], reference_2d[0]], [qp_2d[1], reference_2d[1]],
            color=to_rgba(QP_COLOR, 0.84), linestyle=(0, (2, 2)), linewidth=1.0, zorder=7)
    ax.scatter(*lp_2d, color=LP_COLOR, s=24, zorder=10)
    ax.scatter(*qp_2d, color=QP_COLOR, s=24, zorder=11)
    ax.scatter([0.0], [0.0], color=AXIS, s=18, zorder=12)

    ax.text(0.055, 0.935, "reference", transform=ax.transAxes, color=REFERENCE_COLOR,
            fontsize=9, fontweight="medium")
    ax.text(0.055, 0.885, "LP  ·  colinear", transform=ax.transAxes, color=LP_COLOR,
            fontsize=9, fontweight="medium")
    ax.text(0.055, 0.835, "QP  ·  nearest", transform=ax.transAxes, color=QP_COLOR,
            fontsize=9, fontweight="medium")
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba, mode="RGBA")


def make_animation(output: Path, wheels: int, frames: int, fps: int, pixels: int) -> Path:
    phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    images = gif_frames([render_frame(wheels, phase, pixels) for phase in phases])
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                    loop=0, disposal=2, transparency=255, optimize=False)
    return output


def generate(output_dir: Path, frames: int, fps: int, pixels: int) -> list[Path]:
    if frames < 2 or fps <= 0 or pixels <= 0:
        raise ValueError("frames must be >= 2; fps and pixels must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        make_animation(output_dir / f"3mtq_{wheels}rw_lp_qp_tracking_2d.gif", wheels, frames, fps, pixels)
        for wheels in (1, 2, 3)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--fps", type=int, default=14)
    parser.add_argument("--pixels", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for asset in generate(args.output_dir, args.frames, args.fps, args.pixels):
        print(asset)
