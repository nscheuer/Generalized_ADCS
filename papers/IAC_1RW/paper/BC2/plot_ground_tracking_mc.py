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
PREFIX = "bc2_3mtq_1rw_boston_horizon_to_horizon_mc_10"
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
    mean, sigma = np.mean(errors, axis=0), np.std(errors, axis=0, ddof=1)

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
    ax_error.plot(time, mean, color="#0072B2", label="3 MTQ + 1 RW")
    ax_error.fill_between(
        time, np.maximum(0.0, mean - sigma), mean + sigma,
        color="#0072B2", alpha=0.18, linewidth=0.0, label=r"$\pm 1\sigma$",
    )
    ax_error.axhline(CONNECTION_LIMIT_DEG, color="#D55E00", linestyle="--", linewidth=0.9,
                     label=r"Connection limit (5$^\circ$)")
    ax_error.set_ylabel("Ground-target error [deg]")
    ax_error.set_title("BC2 horizon-to-horizon ground contact (N=10)")
    ax_error.set_ylim(bottom=0.0)
    ax_error.grid(True)
    ax_error.legend(loc="upper right", fontsize=5.8)

    run_ids = np.arange(1, len(contact_s) + 1)
    ax_contact.scatter(run_ids, contact_s / 60.0, color="#0072B2", s=17, zorder=3, label="Usable link")
    ax_contact.axhline(np.mean(visibility_s) / 60.0, color="#555555", linestyle="--", linewidth=0.9,
                       label="Geometric window")
    ax_contact.errorbar(
        len(contact_s) + 1.05, np.mean(contact_s) / 60.0,
        yerr=np.std(contact_s, ddof=1) / 60.0, fmt="D", color="#D55E00",
        markersize=4, capsize=2.5, label=r"Mean $\pm 1\sigma$",
    )
    ax_contact.set_xlim(0.4, len(contact_s) + 1.8)
    ax_contact.set_xticks(run_ids)
    ax_contact.set_xlabel("Monte Carlo run")
    ax_contact.set_ylabel("Link time [min]")
    ax_contact.grid(True)
    ax_contact.legend(loc="lower left", fontsize=5.6)
    ax_contact.text(
        0.98, 0.96,
        "5$^\\circ$ cone, geometric LOS only\n"
        f"min / mean / max: {contact_s.min()/60.0:.2f} / {contact_s.mean()/60.0:.2f} / {contact_s.max()/60.0:.2f} min\n"
        f"$1\\sigma$: {contact_s.std(ddof=1)/60.0:.2f} min",
        transform=ax_contact.transAxes, ha="right", va="top", fontsize=5.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    fig.tight_layout()
    output = OUTPUT_DIR / "bc2_ground_tracking_contact_and_convergence_mc_10.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = OUTPUT_DIR / "bc2_ground_contact_times_mc_10.csv"
    np.savetxt(
        summary,
        np.column_stack((run_ids, visibility_s, contact_s)),
        delimiter=",", header="run_id,geometric_visibility_s,usable_connection_s", comments="", fmt=["%d", "%.1f", "%.1f"],
    )
    print(f"Saved {output}")
    print(f"Saved {summary}")
    print(
        "Usable connection time [s]: "
        f"min={contact_s.min():.1f}, mean={contact_s.mean():.1f}, "
        f"max={contact_s.max():.1f}, sigma={contact_s.std(ddof=1):.1f}"
    )


if __name__ == "__main__":
    main()
