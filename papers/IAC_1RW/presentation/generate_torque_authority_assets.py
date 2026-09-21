#!/usr/bin/env python3
"""Generate the complete transparent 3-MTQ torque-authority asset set.

Outputs are written to ``torque_authority_assets`` beside this script:

* four introductory 3-MTQ PNG stills (dipole cube, field, perpendicular plane,
  and their combination);
* for 1, 2, and 3 reaction wheels: a dipole-cube/RW-envelope still, a moving
  MTQ-plane GIF, and a moving Minkowski-sum GIF.

All assets share the clean, no-chart-furniture visual language used by
``animate_3mtq_torque_polytope.py``.  PNGs retain full alpha; GIFs use GIF's
native binary transparency.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial import ConvexHull

from animate_3mtq_torque_polytope import (
    AXIS,
    CYAN,
    FIELD,
    VIOLET,
    draw_arrow,
    gif_frames,
    magnetic_field,
    plane_basis,
    torque_polytope,
    unit,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "torque_authority_assets"
MTQ_DIPOLE_LIMIT = 0.20  # A m^2 per axis
FIELD_STRENGTH = 42e-6  # T
RW_TORQUE_LIMIT = 1.15e-5  # N m per wheel


def setup_axes(limit: float, pixels: int) -> tuple[plt.Figure, plt.Axes]:
    """Create a transparent, origin-centred 3-D scene without chart furniture."""
    figure = plt.figure(figsize=(5.0, 5.0), dpi=pixels / 5.0)
    figure.patch.set_alpha(0.0)
    ax = figure.add_subplot(111, projection="3d", computed_zorder=False)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=21, azim=-48)
    ax.set_proj_type("persp", focal_length=0.9)

    axis_end = 0.92 * limit
    for vector in np.eye(3):
        ax.plot(
            [-axis_end * vector[0], axis_end * vector[0]],
            [-axis_end * vector[1], axis_end * vector[1]],
            [-axis_end * vector[2], axis_end * vector[2]],
            color=to_rgba(AXIS, 0.72), linewidth=0.72, solid_capstyle="round", zorder=25,
        )
    ax.scatter([0.0], [0.0], [0.0], color=AXIS, s=10, depthshade=False, zorder=40)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    return figure, ax


def finish(figure: plt.Figure) -> Image.Image:
    """Capture a figure as a full-alpha PNG-style image."""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba, mode="RGBA")


def cube_vertices(half_width: float) -> np.ndarray:
    return np.asarray(list(product((-half_width, half_width), repeat=3)), dtype=float)


def draw_wire_cube(ax: plt.Axes, half_width: float, *, color: str = VIOLET, alpha: float = 0.96) -> None:
    """Draw a restrained, translucent dipole-command cube."""
    vertices = cube_vertices(half_width)
    faces = [vertices[[0, 1, 3, 2]], vertices[[4, 5, 7, 6]], vertices[[0, 1, 5, 4]],
             vertices[[2, 3, 7, 6]], vertices[[0, 2, 6, 4]], vertices[[1, 3, 7, 5]]]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=to_rgba(color, 0.18), edgecolor=to_rgba(color, alpha),
        linewidth=1.0, zorder=5,
    ))


def draw_perpendicular_plane(ax: plt.Axes, b_field: np.ndarray, radius: float) -> None:
    """Draw a square plane whose normal is the magnetic-field direction."""
    first, second = plane_basis(b_field)
    corners = np.array((
        -first - second, first - second, first + second, -first + second,
    )) * radius
    ax.add_collection3d(Poly3DCollection(
        [corners], facecolor=to_rgba(CYAN, 0.30), edgecolor=to_rgba(CYAN, 0.92),
        linewidth=1.15, zorder=5,
    ))
    closed = np.vstack((corners, corners[0]))
    ax.plot(*closed.T, color=CYAN, linewidth=1.15, zorder=6)


def rw_vertices(wheels: int, limit: float) -> np.ndarray:
    """Vertices of the reaction-wheel torque box in the first ``wheels`` axes."""
    if wheels not in (1, 2, 3):
        raise ValueError("wheels must be 1, 2, or 3")
    signs = np.asarray(list(product((-1.0, 1.0), repeat=wheels)))
    vertices = np.zeros((len(signs), 3))
    vertices[:, :wheels] = limit * signs
    return vertices


def draw_rw_envelope(
    ax: plt.Axes, wheels: int, limit: float, *, color: str = VIOLET, alpha: float = 0.96,
) -> None:
    """Draw the RW line, rectangle, or cube envelope as a precise wireframe."""
    vertices = rw_vertices(wheels, limit)
    for first, point in enumerate(vertices):
        for second in range(first + 1, len(vertices)):
            # Vertices connected by one sign change form a box edge.
            if np.count_nonzero(np.abs(vertices[first] - vertices[second]) > 0.0) == 1:
                ax.plot(*np.vstack((point, vertices[second])).T, color=to_rgba(color, alpha),
                        linewidth=1.75, solid_capstyle="round", zorder=31)


def draw_mtq_plane(ax: plt.Axes, b_field: np.ndarray, *, alpha: float = 0.86) -> np.ndarray:
    """Draw the physically calculated 3-MTQ torque polygon."""
    polygon = torque_polytope(b_field, MTQ_DIPOLE_LIMIT)
    ax.add_collection3d(Poly3DCollection(
        [polygon], facecolor=to_rgba(CYAN, alpha), edgecolor=to_rgba(CYAN, 0.98),
        linewidth=0.85, zorder=5,
    ))
    ax.plot(*np.vstack((polygon, polygon[0])).T, color=VIOLET, linewidth=1.45,
            solid_capstyle="round", zorder=7)
    return polygon


def draw_minkowski_sum(ax: plt.Axes, b_field: np.ndarray, wheels: int) -> None:
    """Draw conv(MTQ plane vertices + RW box vertices), the combined authority."""
    mtq = torque_polytope(b_field, MTQ_DIPOLE_LIMIT)
    rw = rw_vertices(wheels, RW_TORQUE_LIMIT)
    points = (mtq[:, None, :] + rw[None, :, :]).reshape(-1, 3)
    # A 1-RW sum is exactly planar when the MTQ plane contains that wheel's
    # torque axis.  QJ resolves this instantaneous physical degeneracy with a
    # numerical perturbation small enough to be visually imperceptible, keeping
    # a single continuous GIF rather than failing on that frame.
    hull = ConvexHull(points, qhull_options="QJ")
    triangles = [points[simplex] for simplex in hull.simplices]
    ax.add_collection3d(Poly3DCollection(
        triangles, facecolor=to_rgba(VIOLET, 0.76), edgecolor=to_rgba(CYAN, 0.90),
        linewidth=0.68, zorder=5,
    ))


def render_intro(kind: str, pixels: int) -> Image.Image:
    """Render one of the four foundational 3-MTQ geometry stills."""
    figure, ax = setup_axes(1.8, pixels)
    b_field = unit(np.array((0.54, -0.23, 0.81)))
    if kind in {"cube", "combined"}:
        draw_wire_cube(ax, 0.72)
    if kind in {"field", "plane", "combined"}:
        draw_arrow(ax, b_field, FIELD, length=1.45, width=2.0, zorder=35)
    if kind in {"plane", "combined"}:
        draw_perpendicular_plane(ax, b_field, 0.95)
    return finish(figure)


def render_rw_still(wheels: int, pixels: int) -> Image.Image:
    """Render the dipole cube with its corresponding RW torque envelope."""
    figure, ax = setup_axes(1.8, pixels)
    draw_wire_cube(ax, 0.70, color=CYAN)
    # This is a conceptual authority comparison: both bounded actuator sets
    # share the same centred visual frame, not a claim of shared physical units.
    draw_rw_envelope(ax, wheels, 1.35, color=VIOLET)
    return finish(figure)


def render_torque_animation_frame(kind: str, wheels: int, phase: float, pixels: int) -> Image.Image:
    """Render either the moving plane/RW scene or the moving combined volume."""
    limit = 1.65 * MTQ_DIPOLE_LIMIT * FIELD_STRENGTH + RW_TORQUE_LIMIT
    figure, ax = setup_axes(limit, pixels)
    b_field = magnetic_field(phase, FIELD_STRENGTH)
    if kind == "plane":
        draw_mtq_plane(ax, b_field)
        draw_rw_envelope(ax, wheels, RW_TORQUE_LIMIT)
    elif kind == "minkowski":
        draw_minkowski_sum(ax, b_field, wheels)
        # Retain the two source authorities as colour-coded references within
        # the combined envelope: MTQ cyan, RW violet.
        draw_mtq_plane(ax, b_field, alpha=0.30)
        draw_rw_envelope(ax, wheels, RW_TORQUE_LIMIT, alpha=0.82)
    else:
        raise ValueError(f"Unknown animation kind: {kind}")
    draw_arrow(ax, b_field, FIELD, length=0.78 * limit, width=1.9, zorder=35)
    ax.scatter([0.0], [0.0], [0.0], color=AXIS, s=10, depthshade=False, zorder=40)
    return finish(figure)


def save_gif(path: Path, kind: str, wheels: int, frames: int, fps: int, pixels: int) -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    rendered_frames = [render_torque_animation_frame(kind, wheels, phase, pixels) for phase in phases]
    images = gif_frames(rendered_frames)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
        disposal=2, transparency=255, optimize=False,
    )


def generate(output: Path, *, frames: int, fps: int, pixels: int) -> list[Path]:
    """Create every requested still and animation, returning their paths."""
    if frames < 2 or fps <= 0 or pixels <= 0:
        raise ValueError("frames must be >= 2; fps and pixels must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    intro = (
        ("01_3mtq_magnetic_moment_cube", "cube"),
        ("02_magnetic_field_line", "field"),
        ("03_magnetic_field_perpendicular_plane", "plane"),
        ("04_3mtq_geometry_combined", "combined"),
    )
    for stem, kind in intro:
        path = output / f"{stem}.png"
        render_intro(kind, pixels).save(path)
        paths.append(path)

    for wheels in (1, 2, 3):
        prefix = f"{wheels}rw"
        still = output / f"{prefix}_01_moment_cube_rw_envelope.png"
        render_rw_still(wheels, pixels).save(still)
        paths.append(still)
        for number, kind, description in (
            ("02", "plane", "moving_mtq_plane"),
            ("03", "minkowski", "moving_minkowski_sum"),
        ):
            path = output / f"{prefix}_{number}_{description}.gif"
            save_gif(path, kind, wheels, frames, fps, pixels)
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=72, help="Frames per looping animation.")
    parser.add_argument("--fps", type=int, default=14, help="Slow, presentation-friendly frame rate.")
    parser.add_argument("--pixels", type=int, default=960, help="Square raster dimension for every asset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for asset in generate(args.output_dir, frames=args.frames, fps=args.fps, pixels=args.pixels):
        print(asset)
