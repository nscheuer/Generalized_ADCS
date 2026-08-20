#!/usr/bin/env python3
"""Compare Boston ground tracking with and without LOS-rate feed-forward."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ADCS  # noqa: E402
from ADCS.helpers.math_helpers import rot_mat  # noqa: E402


BOSTON_LAT_DEG = 42.36
BOSTON_LON_DEG = -71.06
SIM_DURATION_S = 250.0
SIM_DT_S = 2.0
TAIL_DURATION_S = 80.0

TRACKING_COLOR = "#0072B2"
NO_RATE_COLOR = "#D55E00"
GRID_COLOR = "#C7CDD3"


class DirectionOnlyCoordinateGoal(ADCS.goals.Coordinate_Goal):
    """Track the moving LOS direction without its angular-rate feed-forward."""

    def to_ref(self, os0):
        target, _w_ref_eci = super().to_ref(os0)
        return target, np.zeros(3)


def make_initial_orbit() -> ADCS.Orbital_State:
    """Use the Boston-tracking orbit from the framework tutorial."""
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


def run_case(goal) -> ADCS.SimulationResults:
    """Simulate one deterministic 3 MTQ + 3 RW ground-tracking case."""
    np.random.seed(7)
    satellite = ADCS.satellite_factory.create_3_3_beavercube2_cubesat(
        estimated=False
    )
    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5.0e-5,
        d_gain=2.0e-3,
        c_gain=1.0e-3,
        h_target=np.zeros(3),
    )
    initial_state = np.array(
        [0.01, -0.02, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    return ADCS.simulate(
        x=initial_state,
        satellite=satellite,
        controller=controller,
        goal=goal,
        os0=make_initial_orbit(),
        dt=SIM_DT_S,
        tf=SIM_DURATION_S,
    )


def tracking_error_deg(results: ADCS.SimulationResults) -> np.ndarray:
    """Return boresight-to-target angular error for every recorded sample."""
    run = results.first()
    states = np.asarray(run.state_hist, dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)[:, 1:4]
    boresights = np.asarray(run.boresight_hist, dtype=float)

    rotations = np.asarray([rot_mat(q) for q in states[:, 3:7]])
    boresight_eci = np.einsum("nij,nj->ni", rotations, boresights)
    boresight_eci /= np.linalg.norm(boresight_eci, axis=1, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    dots = np.einsum("ni,ni->n", boresight_eci, targets)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def save_tracking_error_plot(
    rate_results: ADCS.SimulationResults,
    no_rate_results: ADCS.SimulationResults,
) -> tuple[Path, Path]:
    """Save the compact PNG/PDF tracking-error comparison."""
    time_s = np.asarray(rate_results.first().time_s, dtype=float)
    rate_error = tracking_error_deg(rate_results)
    no_rate_error = tracking_error_deg(no_rate_results)
    tail = time_s >= time_s[-1] - TAIL_DURATION_S
    rate_tail_mean = float(np.mean(rate_error[tail]))
    no_rate_tail_mean = float(np.mean(no_rate_error[tail]))

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)
    ax.axvspan(
        time_s[-1] - TAIL_DURATION_S,
        time_s[-1],
        color="#EEF1F4",
        zorder=0,
    )
    ax.plot(
        time_s,
        no_rate_error,
        color=NO_RATE_COLOR,
        linewidth=1.5,
        label=r"LOS only ($\omega_{ref}=0$)",
        zorder=2,
    )
    ax.plot(
        time_s,
        rate_error,
        color=TRACKING_COLOR,
        linewidth=1.8,
        label=r"LOS + $\omega_{ref}$ feed-forward",
        zorder=3,
    )
    ax.set_title("Boston Ground-Target Tracking")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pointing error [deg]")
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.06,
        "Final 80 s mean\n"
        + rf"{rate_tail_mean:.1f}$^\circ$ with rate"
        + "\n"
        + rf"{no_rate_tail_mean:.1f}$^\circ$ without",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#30343B",
    )

    png_path = SCRIPT_DIR / "ground_tracking_error.png"
    pdf_path = SCRIPT_DIR / "ground_tracking_error.pdf"
    fig.savefig(png_path, dpi=600, transparent=True)
    fig.savefig(pdf_path, transparent=True)
    plt.close(fig)
    return png_path, pdf_path


def raster_png_to_pdf(png_path: Path, pdf_path: Path, dpi: float = 300.0) -> None:
    """Place a PyVista snapshot into a borderless, same-aspect PDF page."""
    image = plt.imread(png_path)
    height_px, width_px = image.shape[:2]
    fig = plt.figure(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
        frameon=False,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def crop_satellite_quadrant(source_path: Path, output_path: Path) -> None:
    """Enlarge the North America quadrant while preserving the globe view."""
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        width, height = source.size
        crop_box = (
            round(0.15 * width),
            round(0.07 * height),
            round(0.65 * width),
            round(0.57 * height),
        )
        quadrant = source.crop(crop_box)
        poster_image = ImageOps.fit(
            quadrant,
            (1100, 800),
            method=Image.Resampling.LANCZOS,
        )

    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, size=25)
    draw = ImageDraw.Draw(poster_image)
    draw.text(
        (10, 8),
        "3 MTQ + 3 RW satellite tracking Boston",
        fill="black",
        font=font,
    )
    poster_image.save(output_path)


def save_animation_snapshot(
    results: ADCS.SimulationResults,
    goal: ADCS.goals.Coordinate_Goal,
) -> tuple[Path, Path]:
    """Save the final frame from the framework's AnimationPlot."""
    png_path = SCRIPT_DIR / "ground_tracking_3d.png"
    pdf_path = SCRIPT_DIR / "ground_tracking_3d.pdf"
    raw_path = SCRIPT_DIR / ".ground_tracking_3d_full.png"
    animation = ADCS.plots.AnimationPlot(
        goal=goal,
        title="",
        window_size=(1800, 1300),
        smooth_factor=2,
        min_smooth_N=500,
        show_est_orbit=False,
        show_env_vectors=False,
        axis_scale_body=0.22,
        axis_scale_goal=0.55,
        satellite_marker_scale=0.009,
        goal_marker_scale=0.025,
        camera_zoom=1.15,
        background_color="white",
        text_color="black",
    )
    animation.save_snapshot(results, raw_path, frame_index=-1)
    crop_satellite_quadrant(raw_path, png_path)
    raw_path.unlink()
    raster_png_to_pdf(png_path, pdf_path)
    return png_path, pdf_path


def main() -> None:
    rate_goal = ADCS.goals.Coordinate_Goal(
        lat=BOSTON_LAT_DEG,
        lon=BOSTON_LON_DEG,
        alt=0.0,
    )
    no_rate_goal = DirectionOnlyCoordinateGoal(
        lat=BOSTON_LAT_DEG,
        lon=BOSTON_LON_DEG,
        alt=0.0,
    )

    print("Simulating tracking with LOS angular-rate feed-forward...")
    rate_results = run_case(rate_goal)
    print("Simulating direction-only tracking...")
    no_rate_results = run_case(no_rate_goal)

    outputs = [
        *save_animation_snapshot(rate_results, rate_goal),
        *save_tracking_error_plot(rate_results, no_rate_results),
    ]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
