"""Plot horizon-to-horizon tracking and usable ground-contact time."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from plot_style import configure_ieee_style


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PREFIX = "bc2_3mtq_1rw_boston_horizon_to_horizon_mc_100"
BOSTON_LAT_DEG = 42.36
BOSTON_LON_DEG = -71.06
CONNECTION_LIMIT_DEG = 5.0


def _latest_sim() -> Path:
    paths = list(OUTPUT_DIR.glob(f"{PREFIX}_*.sim"))
    if not paths:
        raise FileNotFoundError(f"No archive matching {PREFIX!r} in {OUTPUT_DIR}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _error_deg(run: ADCS.RunResults) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([state.q for state in run.state_hist], dtype=float)
    target = np.asarray(run.target_hist, dtype=float)
    target = target[:, 1:4] if target.shape[1] == 4 else target
    boresight = np.asarray(run.boresight_hist, dtype=float)
    boresight_eci = np.einsum("nij,nj->ni", np.asarray([rot_mat(x) for x in q]), boresight)
    boresight_eci /= np.linalg.norm(boresight_eci, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    return np.asarray(run.time_s, dtype=float), np.rad2deg(
        np.arccos(np.clip(np.sum(boresight_eci * target, axis=1), -1.0, 1.0))
    )


def _geometrically_visible(run: ADCS.RunResults, goal: ADCS.goals.Coordinate_Goal) -> np.ndarray:
    """Return the horizon mask; refraction and atmospheric losses are ignored."""
    ephem = ADCS.Ephemeris()
    orbital_states = [
        ADCS.Orbital_State.from_dict(os, ephem) if isinstance(os, dict) else os
        for os in run.os_hist
    ]
    target_eci = np.asarray([os.ecef_to_eci(goal.target_ecef) for os in orbital_states])
    spacecraft_eci = np.asarray([os.R for os in orbital_states])
    # LOS has positive elevation at the target when it points outward from the
    # local horizon plane.  This is the purely geometric AOS/LOS criterion.
    return np.einsum("ij,ij->i", spacecraft_eci - target_eci, target_eci) > 0.0


def _duration_s(time: np.ndarray, active: np.ndarray) -> float:
    """Integrate complete sample intervals satisfying a Boolean condition."""
    return float(np.sum(np.diff(time)[active[:-1] & active[1:]]))


def main() -> None:
    configure_ieee_style()
    results = ADCS.SimulationResults.load(_latest_sim())
    goal = ADCS.goals.Coordinate_Goal(BOSTON_LAT_DEG, BOSTON_LON_DEG, 0.0)
    traces = [_error_deg(run) for run in results]
    time = traces[0][0]
    errors = np.vstack([np.interp(time, t, error) for t, error in traces])
    visible = np.vstack([_geometrically_visible(run, goal) for run in results])
    connected = visible & (errors <= CONNECTION_LIMIT_DEG)
    contact_s = np.asarray([_duration_s(time, mask) for mask in connected])
    visibility_s = np.asarray([_duration_s(time, mask) for mask in visible])
    p10, median, p90 = np.percentile(errors, (10, 50, 90), axis=0)
    contact_p10, contact_median, contact_p90 = np.percentile(
        contact_s, (10, 50, 90)
    )

    # Orbit and target are common to all trials, so the geometric horizon is
    # common too.  Use it to annotate the physically available contact window.
    horizon = visible[0]
    horizon_idx = np.flatnonzero(horizon)
    aos_s, los_s = time[horizon_idx[0]], time[horizon_idx[-1]]

    fig, (ax_error, ax_contact) = plt.subplots(
        2, 1, figsize=(3.45, 4.25), sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_error.axvspan(aos_s, los_s, color="#009E73", alpha=0.08, label="Geometric visibility")
    ax_error.plot(time, median, color="#0072B2", label="Median")
    ax_error.fill_between(
        time, p10, p90,
        color="#0072B2", alpha=0.18, linewidth=0.0, label="10--90 percentile",
    )
    ax_error.axhline(CONNECTION_LIMIT_DEG, color="#D55E00", linestyle="--", linewidth=0.9,
                     label=r"Connection limit (5$^\circ$)")
    ax_error.set_ylabel("Ground-target error [deg]")
    ax_error.set_title("BC2 ground contact (median, 10--90%, N=100)")
    ax_error.set_ylim(bottom=0.0)
    ax_error.grid(True)
    ax_error.legend(loc="upper right", fontsize=5.8)

    contact_minutes = contact_s / 60.0
    horizon_minutes = np.mean(visibility_s) / 60.0
    ax_contact.axvspan(
        contact_p10 / 60.0, contact_p90 / 60.0,
        color="#0072B2", alpha=0.16, linewidth=0.0, label="10--90%",
    )
    ax_contact.hist(
        contact_minutes, bins=12, color="#0072B2", alpha=0.82,
        edgecolor="white", linewidth=0.5, label="Runs",
    )
    ax_contact.axvline(
        contact_median / 60.0, color="#D55E00", linewidth=1.0,
        label="Median",
    )
    ax_contact.axvline(
        horizon_minutes, color="#555555", linestyle="--", linewidth=0.9,
        label="Horizon",
    )
    ax_contact.set_xlabel("Usable link time [min]")
    ax_contact.set_ylabel("Number of runs")
    ax_contact.set_title("Usable link-time distribution")
    ax_contact.grid(True)
    ax_contact.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
        fontsize=5.6, frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    output = OUTPUT_DIR / "bc2_ground_tracking_contact_and_convergence_mc_100.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = OUTPUT_DIR / "bc2_ground_contact_times_mc_100.csv"
    run_ids = np.arange(1, len(contact_s) + 1)
    np.savetxt(
        summary,
        np.column_stack((run_ids, visibility_s, contact_s)),
        delimiter=",", header="run_id,geometric_visibility_s,usable_connection_s", comments="", fmt=["%d", "%.1f", "%.1f"],
    )
    print(f"Saved {output}")
    print(f"Saved {summary}")
    print(
        "Usable connection time [s]: "
        f"min={contact_s.min():.1f}, median={contact_median:.1f}, "
        f"max={contact_s.max():.1f}, p10={contact_p10:.1f}, p90={contact_p90:.1f}"
    )


if __name__ == "__main__":
    main()
