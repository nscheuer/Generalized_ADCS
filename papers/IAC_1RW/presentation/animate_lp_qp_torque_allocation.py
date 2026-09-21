#!/usr/bin/env python3
"""Animate LP and QP torque allocation for 3 MTQ + 1/2/3 RW clusters.

The magnetic field is deliberately *not* drawn.  It remains fixed internally
to define the MTQ torque map, while the viewer sees only the resulting
authority envelope, a moving reference torque, and the two allocator outputs:

* LP (green): maximum feasible torque exactly colinear with the reference.
* QP (orange): feasible torque whose endpoint is nearest in Euclidean distance.

The formulations mirror ``MTQ_w_RW_LP.allocate_max_torque_in_direction`` and
``MTQ_w_RW_QP.allocate_max_torque_in_direction`` in the ADCS source tree.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.optimize import linprog, lsq_linear
from scipy.spatial import ConvexHull

from animate_3mtq_torque_polytope import AXIS, CYAN, VIOLET, draw_arrow, gif_frames
from generate_torque_authority_assets import finish, setup_axes


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "allocator_tracking_assets"
MTQ_DIPOLE_LIMIT = 0.20  # A m^2
RW_TORQUE_LIMIT = 1.15e-5  # N m
# This field defines the MTQ map but is intentionally absent from every frame.
B_BODY = np.array((22e-6, -8e-6, 41e-6))
LP_COLOR = "#22C55E"
QP_COLOR = "#F97316"
REFERENCE_COLOR = "#334155"


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the matrix [vector]_x such that [v]_x w = v x w."""
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def actuator_map(wheels: int) -> tuple[np.ndarray, np.ndarray]:
    """Build A_total and symmetric command bounds for a 3-MTQ + N-RW cluster."""
    if wheels not in (1, 2, 3):
        raise ValueError("wheels must be 1, 2, or 3")
    rw_map = np.eye(3)[:, :wheels]
    mtq_map = -skew(B_BODY) @ np.eye(3)
    matrix = np.hstack((rw_map, mtq_map))
    limits = np.r_[np.full(wheels, RW_TORQUE_LIMIT), np.full(3, MTQ_DIPOLE_LIMIT)]
    return matrix, limits


def lp_solution(matrix: np.ndarray, limits: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Project along the requested ray using the repository LP formulation."""
    magnitude = np.linalg.norm(reference)
    direction = reference / magnitude
    # Optimize normalized commands v = u / u_max.  This is algebraically the
    # same LP used by the controller, but avoids mixing 1e-5 N m RW commands
    # and 1e-1 A m^2 MTQ commands in one ill-conditioned decision vector.
    normalized_map = matrix * limits
    torque_scale = 1e-5
    # min -T, subject to A u - T tau_hat = 0 and bounded actuator commands.
    objective = np.zeros(normalized_map.shape[1] + 1)
    objective[-1] = -1.0
    # The available-torque variable is likewise normalized so HiGHS evaluates
    # equality feasibility at O(1), rather than at magnetic-torque magnitudes.
    equality = np.column_stack((normalized_map / torque_scale, -direction))
    bounds = [(-1.0, 1.0) for _ in limits] + [(0.0, None)]
    result = linprog(objective, A_eq=equality, b_eq=np.zeros(3), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP allocation failed: {result.message}")
    available = result.x[-1] * torque_scale
    # The controller scales down if the reference lies inside the envelope.
    normalized_commands = result.x[:-1] * min(1.0, magnitude / available)
    return normalized_map @ normalized_commands


def qp_solution(matrix: np.ndarray, limits: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the bounded least-squares endpoint from the repository QP allocator."""
    # This is min ||A diag(u_max) v - tau_ref||², |v_i| <= 1, which is exactly
    # the source QP under u = diag(u_max) v but with well-scaled variables.
    normalized_map = matrix * limits
    result = lsq_linear(normalized_map, reference, bounds=(-1.0, 1.0), method="trf", tol=1e-14)
    if not result.success:
        raise RuntimeError(f"QP allocation failed: {result.message}")
    return normalized_map @ result.x


def authority_points(matrix: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Map every command-box corner to torque space."""
    commands = np.asarray(list(product((-1.0, 1.0), repeat=len(limits)))) * limits
    return commands @ matrix.T


def reference_torque(phase: float) -> np.ndarray:
    """Smooth, always-infeasible torque request that exposes both allocators."""
    direction = np.array((
        0.79 * np.cos(phase) + 0.15 * np.sin(2.0 * phase),
        0.67 * np.sin(phase),
        0.46 * np.sin(2.0 * phase + 0.45),
    ))
    # Intentionally outside the 3+3 hull as well, so every configuration
    # exposes the distinction between directional and nearest-point tracking.
    return 4.10e-5 * direction / np.linalg.norm(direction)


def draw_authority(ax, points: np.ndarray) -> None:
    """Draw the static combined torque hull with a subtle cyan-violet treatment."""
    hull = ConvexHull(points, qhull_options="QJ")
    faces = [points[simplex] for simplex in hull.simplices]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=to_rgba(VIOLET, 0.42), edgecolor=to_rgba(CYAN, 0.72),
        linewidth=0.55, zorder=4,
    ))


def draw_solution(ax, reference: np.ndarray, lp: np.ndarray, qp: np.ndarray) -> None:
    """Draw the reference and allocator arrows plus the nearest-point residual."""
    # Reference: a restrained dashed ray and dark arrow.  There is no B-field arrow.
    ax.plot(*np.vstack((np.zeros(3), reference)).T, color=to_rgba(REFERENCE_COLOR, 0.52),
            linewidth=1.05, linestyle=(0, (2.2, 2.6)), zorder=30)
    draw_arrow(ax, reference, REFERENCE_COLOR, length=np.linalg.norm(reference), width=1.55, zorder=31)

    # LP is perfectly colinear by construction.  QP uses a visible connector to
    # make its shortest Euclidean residual immediately apparent.
    draw_arrow(ax, lp, LP_COLOR, length=np.linalg.norm(lp), width=2.3, zorder=34)
    draw_arrow(ax, qp, QP_COLOR, length=np.linalg.norm(qp), width=2.3, zorder=35)
    ax.plot(*np.vstack((qp, reference)).T, color=to_rgba(QP_COLOR, 0.82), linewidth=1.0,
            linestyle=(0, (1.6, 2.0)), zorder=33)
    ax.scatter(*lp, color=LP_COLOR, s=20, depthshade=False, zorder=36)
    ax.scatter(*qp, color=QP_COLOR, s=20, depthshade=False, zorder=37)
    # A compact key makes the allocation distinction self-explanatory without
    # adding conventional chart furniture or a field-vector annotation.
    ax.text2D(0.055, 0.935, "reference", transform=ax.transAxes, color=REFERENCE_COLOR,
              fontsize=9, fontweight="medium")
    ax.text2D(0.055, 0.885, "LP  ·  colinear", transform=ax.transAxes, color=LP_COLOR,
              fontsize=9, fontweight="medium")
    ax.text2D(0.055, 0.835, "QP  ·  nearest", transform=ax.transAxes, color=QP_COLOR,
              fontsize=9, fontweight="medium")


def render_frame(wheels: int, phase: float, pixels: int) -> Image.Image:
    """Render one allocator comparison frame; the field is never visualized."""
    matrix, limits = actuator_map(wheels)
    points = authority_points(matrix, limits)
    reference = reference_torque(phase)
    lp = lp_solution(matrix, limits, reference)
    qp = qp_solution(matrix, limits, reference)
    figure, ax = setup_axes(4.70e-5, pixels)
    draw_authority(ax, points)
    draw_solution(ax, reference, lp, qp)
    ax.scatter([0.0], [0.0], [0.0], color=AXIS, s=10, depthshade=False, zorder=40)
    return finish(figure)


def make_animation(output: Path, wheels: int, frames: int, fps: int, pixels: int) -> Path:
    """Create one seamless, transparent LP-vs-QP allocation animation."""
    phases = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    rendered = [render_frame(wheels, phase, pixels) for phase in phases]
    images = gif_frames(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                   loop=0, disposal=2, transparency=255, optimize=False)
    return output


def generate(output_dir: Path, frames: int, fps: int, pixels: int) -> list[Path]:
    """Generate the requested 3+1, 3+2, and 3+3 allocator GIFs."""
    if frames < 2 or fps <= 0 or pixels <= 0:
        raise ValueError("frames must be >= 2; fps and pixels must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        make_animation(output_dir / f"3mtq_{wheels}rw_lp_qp_tracking.gif", wheels, frames, fps, pixels)
        for wheels in (1, 2, 3)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=84, help="Frames per seamless loop.")
    parser.add_argument("--fps", type=int, default=14, help="Presentation-friendly frame rate.")
    parser.add_argument("--pixels", type=int, default=960, help="Square output resolution.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for asset in generate(args.output_dir, args.frames, args.fps, args.pixels):
        print(asset)
