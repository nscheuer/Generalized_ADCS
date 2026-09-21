#!/usr/bin/env python3
"""Render a transparent GIF of a 3-MTQ torque polytope in a varying B field.

The envelope is the image of the bounded dipole cube under ``tau = m x B``.
It therefore always lies in the plane perpendicular to the instantaneous
magnetic-field vector.  The scene intentionally has no chart furniture: the
three small axes at the origin are the only reference frame.

Example
-------
    venv/bin/python papers/IAC_1RW/presentation/animate_3mtq_torque_polytope.py

The resulting GIF has a binary transparent background, which is the highest
transparency fidelity supported by the GIF format.  For use cases requiring
soft, partial transparency, render to a video or PNG sequence instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "3mtq_torque_polytope.gif"

# A restrained, high-contrast palette that survives placement over light or
# dark slides.  The transparent canvas makes the animation presentation-ready.
AXIS = "#64748B"
CYAN = "#06B6D4"
VIOLET = "#7C3AED"
FIELD = "#F97316"


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, rejecting a degenerate magnetic field."""
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("The magnetic field must be non-zero.")
    return vector / norm


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal basis for the plane normal to ``normal``."""
    normal = unit(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = unit(np.cross(normal, reference))
    return first, unit(np.cross(normal, first))


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2-D hull using Andrew's monotonic chain."""
    unique_points = sorted(set(map(tuple, np.round(points, 15))))

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def torque_polytope(b_field: np.ndarray, dipole_limit: float) -> np.ndarray:
    """Return the vertices of the bounded 3-MTQ torque envelope.

    Each MTQ is independently bounded by +/- ``dipole_limit`` A m^2.  The
    eight corners of that dipole cube are mapped through the cross product and
    then reduced to their convex polygon in the torque plane.
    """
    dipoles = np.array(np.meshgrid(*[[-dipole_limit, dipole_limit]] * 3)).T.reshape(-1, 3)
    torque_corners = np.cross(dipoles, b_field)
    center = torque_corners.mean(axis=0)
    first, second = plane_basis(b_field)
    projected = np.column_stack(((torque_corners - center) @ first, (torque_corners - center) @ second))
    hull = convex_hull_2d(projected)
    return center + hull[:, :1] * first + hull[:, 1:] * second


def magnetic_field(phase: float, field_strength: float) -> np.ndarray:
    """A smooth, closed body-frame field trajectory with modest magnitude drift."""
    direction = np.array((
        0.54 * np.cos(phase) + 0.16 * np.sin(2.0 * phase),
        0.38 * np.sin(phase) - 0.21 * np.cos(2.0 * phase),
        0.79 + 0.15 * np.sin(phase + 0.5),
    ))
    magnitude = field_strength * (1.0 + 0.09 * np.sin(phase - 0.35))
    return magnitude * unit(direction)


def style_axes(ax: plt.Axes, limit: float) -> None:
    """Leave only a compact coordinate triad floating at the origin."""
    ax.set_axis_off()
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    axis_end = 0.92 * limit
    for vector in np.eye(3):
        ax.plot(
            [-axis_end * vector[0], axis_end * vector[0]],
            [-axis_end * vector[1], axis_end * vector[1]],
            [-axis_end * vector[2], axis_end * vector[2]],
            color=to_rgba(AXIS, 0.72), linewidth=0.72, solid_capstyle="round", zorder=25,
        )
    ax.scatter([0.0], [0.0], [0.0], color=AXIS, s=9, depthshade=False, zorder=26)


def draw_arrow(ax: plt.Axes, vector: np.ndarray, color: str, *, length: float, width: float, zorder: int) -> None:
    """Draw a 3-D arrow with a consistent visual length."""
    direction = unit(vector) * length
    ax.quiver(
        0.0, 0.0, 0.0, *direction,
        color=color, linewidth=width, arrow_length_ratio=0.13,
        pivot="tail", normalize=False, zorder=zorder,
    )


def render_frame(
    b_field: np.ndarray,
    dipole_limit: float,
    limit: float,
    pixels: int,
) -> Image.Image:
    """Render one anti-aliased RGBA frame with no canvas/background panel."""
    figure = plt.figure(figsize=(5.0, 5.0), dpi=pixels / 5.0)
    figure.patch.set_alpha(0.0)
    # Explicit artist order keeps the compact coordinate triad readable through
    # the polygon, as is customary in polished explanatory motion graphics.
    ax = figure.add_subplot(111, projection="3d", computed_zorder=False)
    ax.patch.set_alpha(0.0)
    style_axes(ax, limit)
    ax.view_init(elev=21, azim=-48)
    ax.set_proj_type("persp", focal_length=0.9)

    polygon = torque_polytope(b_field, dipole_limit)
    # A saturated face and a violet rim give the polytope depth without
    # introducing chart-like shading or a background panel.
    core = Poly3DCollection(
        [polygon], facecolor=to_rgba(CYAN, 0.88), edgecolor=to_rgba(CYAN, 0.95),
        linewidth=0.8, zorder=5,
    )
    ax.add_collection3d(core)
    closed = np.vstack((polygon, polygon[0]))
    ax.plot(*closed.T, color=VIOLET, linewidth=1.55, solid_capstyle="round", zorder=6)

    draw_arrow(ax, b_field, FIELD, length=0.78 * limit, width=1.9, zorder=35)
    # Re-draw the origin after the plane and field vector so the central triad
    # remains crisp in every camera projection.
    ax.scatter([0.0], [0.0], [0.0], color=AXIS, s=10, depthshade=False, zorder=40)

    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba, mode="RGBA")


def gif_frames(images: list[Image.Image]) -> list[Image.Image]:
    """Convert RGBA frames using one shared palette and one transparent index.

    GIF does not support fractional alpha.  More importantly, some slideware
    and browsers mishandle GIFs containing per-frame colour tables whose
    transparency entry changes.  A single palette with index 255 permanently
    reserved for transparency avoids those full-canvas magenta flashes.
    """
    if not images:
        raise ValueError("At least one frame is required.")

    keyed_frames: list[tuple[Image.Image, np.ndarray]] = []
    for image in images:
        rgba = np.asarray(image).copy()
        transparent = rgba[..., 3] < 32
        rgba[transparent] = (255, 0, 255, 0)  # Reserved key colour, never visible artwork.
        rgba[~transparent, 3] = 255
        keyed_frames.append((Image.fromarray(rgba, mode="RGBA").convert("RGB"), transparent))

    # Quantize only 255 colours so palette entry 255 can never be assigned to
    # artwork.  Every subsequent frame is mapped through this exact palette.
    palette_seed = keyed_frames[0][0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = palette_seed.getpalette()[:768]
    palette += [0] * (768 - len(palette))
    palette[765:768] = [255, 0, 255]
    master_palette = Image.new("P", (1, 1))
    master_palette.putpalette(palette)

    converted_frames: list[Image.Image] = []
    for keyed, transparent in keyed_frames:
        indexed = keyed.quantize(palette=master_palette, dither=Image.Dither.NONE)
        indices = np.asarray(indexed).copy()
        indices[transparent] = 255
        frame = Image.fromarray(indices, mode="P")
        frame.putpalette(palette)
        frame.info["transparency"] = 255
        converted_frames.append(frame)
    return converted_frames


def gif_frame(image: Image.Image) -> Image.Image:
    """Backward-compatible single-frame wrapper for :func:`gif_frames`."""
    return gif_frames([image])[0]


def make_gif(
    output: Path,
    dipole_limit: float,
    field_strength: float,
    frames: int,
    fps: int,
    pixels: int,
) -> Path:
    """Generate a seamless looping torque-polytope GIF."""
    if frames < 2:
        raise ValueError("--frames must be at least 2.")
    if fps <= 0 or pixels <= 0 or dipole_limit <= 0.0 or field_strength <= 0.0:
        raise ValueError("frames, fps, pixels, dipole limit, and field strength must be positive.")

    # The limit covers the largest torque over the animated magnetic field path
    # with enough clear space for the field arrow and the origin-centred axes.
    limit = 1.65 * dipole_limit * field_strength
    phase_values = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    rendered_frames = [
        render_frame(magnetic_field(phase, field_strength), dipole_limit, limit, pixels)
        for phase in phase_values
    ]
    images = gif_frames(rendered_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        transparency=255,
        optimize=False,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=72, help="Number of frames in the seamless loop.")
    parser.add_argument("--fps", type=int, default=14, help="Animation frame rate.")
    parser.add_argument("--pixels", type=int, default=960, help="Square output resolution in pixels.")
    parser.add_argument("--m-max", type=float, default=0.20, help="Per-axis MTQ dipole limit [A m^2].")
    parser.add_argument("--b-strength", type=float, default=42e-6, help="Nominal magnetic-field magnitude [T].")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rendered = make_gif(args.output, args.m_max, args.b_strength, args.frames, args.fps, args.pixels)
    print(rendered)
