#!/usr/bin/env python3
"""Create one clean 2-D LP/QP allocation animation.

This is intentionally an abstract 2-D explainer rather than a cut through a
3-D torque polytope: a cyan/violet square rotates, translates slightly, and
breathes in size while a fixed reference torque is allocated by the two rules.
LP follows the reference ray to the square boundary; QP chooses the closest
point on the square.
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
from PIL import Image

from animate_3mtq_torque_polytope import AXIS, CYAN, VIOLET, gif_frames
from animate_lp_qp_torque_allocation import LP_COLOR, QP_COLOR, REFERENCE_COLOR


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "allocator_tracking_assets" / "lp_qp_tracking_2d_square.gif"
REFERENCE = np.array((3.75e-5, 2.20e-5))
VIEW_LIMIT = 4.85e-5


def square_geometry(phase: float) -> np.ndarray:
    """Return a rotating, translating, gently breathing square."""
    half_width = 1.85e-5 * (1.0 + 0.09 * np.sin(phase + 0.35))
    angle = 0.72 * phase + 0.12 * np.sin(phase)
    center = 0.28e-5 * np.array((np.cos(phase + 0.45), np.sin(phase + 0.45)))
    base = half_width * np.array(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    return base @ rotation.T + center


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    edge = end - start
    fraction = np.clip(np.dot(point - start, edge) / np.dot(edge, edge), 0.0, 1.0)
    return start + fraction * edge


def closest_point_on_square(point: np.ndarray, square: np.ndarray) -> np.ndarray:
    candidates = [
        closest_point_on_segment(point, start, end)
        for start, end in zip(square, np.roll(square, -1, axis=0))
    ]
    return min(candidates, key=lambda candidate: np.linalg.norm(candidate - point))


def ray_boundary(direction: np.ndarray, square: np.ndarray) -> np.ndarray:
    """Return the first positive intersection of the origin ray and square."""
    direction = direction / np.linalg.norm(direction)
    best = None
    for start, end in zip(square, np.roll(square, -1, axis=0)):
        edge = end - start
        matrix = np.column_stack((direction, -edge))
        determinant = np.linalg.det(matrix)
        if abs(determinant) < 1e-18:
            continue
        distance, fraction = np.linalg.solve(matrix, start)
        if distance >= 0.0 and 0.0 <= fraction <= 1.0 and (best is None or distance < best):
            best = distance
    if best is None:
        raise RuntimeError("Reference ray did not intersect the moving square.")
    return best * direction


def arrow(ax: plt.Axes, endpoint: np.ndarray, color: str, width: float, zorder: int) -> None:
    ax.add_patch(FancyArrowPatch(
        (0.0, 0.0), endpoint, arrowstyle="-|>", mutation_scale=14,
        linewidth=width, color=color, shrinkA=0.0, shrinkB=0.0, zorder=zorder,
    ))


def render_frame(phase: float, pixels: int) -> Image.Image:
    square = square_geometry(phase)
    reference = REFERENCE
    lp = ray_boundary(reference, square)
    qp = closest_point_on_square(reference, square)

    figure, ax = plt.subplots(figsize=(5.0, 5.0), dpi=pixels / 5.0)
    figure.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
    ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.add_patch(Polygon(
        square, closed=True, facecolor=to_rgba(CYAN, 0.32),
        edgecolor=to_rgba(VIOLET, 0.98), linewidth=2.2, zorder=2,
    ))
    ax.axhline(0.0, color=to_rgba(AXIS, 0.48), linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=to_rgba(AXIS, 0.48), linewidth=0.8, zorder=1)

    ax.plot([0.0, reference[0]], [0.0, reference[1]], color=to_rgba(REFERENCE_COLOR, 0.48),
            linestyle=(0, (3, 3)), linewidth=1.1, zorder=5)
    arrow(ax, reference, REFERENCE_COLOR, 1.55, 6)
    arrow(ax, lp, LP_COLOR, 2.8, 8)
    arrow(ax, qp, QP_COLOR, 2.8, 9)
    ax.plot([qp[0], reference[0]], [qp[1], reference[1]], color=to_rgba(QP_COLOR, 0.84),
            linestyle=(0, (2, 2)), linewidth=1.0, zorder=7)
    ax.scatter(*lp, color=LP_COLOR, s=24, zorder=10)
    ax.scatter(*qp, color=QP_COLOR, s=24, zorder=11)
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


def make_animation(output: Path, frames: int, fps: int, pixels: int) -> Path:
    phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    images = gif_frames([render_frame(phase, pixels) for phase in phases])
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                    loop=0, disposal=2, transparency=255, optimize=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--fps", type=int, default=14)
    parser.add_argument("--pixels", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.frames < 2 or args.fps <= 0 or args.pixels <= 0:
        raise ValueError("frames must be >= 2; fps and pixels must be positive.")
    print(make_animation(args.output, args.frames, args.fps, args.pixels))
