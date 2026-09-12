"""Plot physical RW/MTQ sizing trade-offs for 3+1 magnetic control.

The contours are geometry-only rejection probabilities.  They use a
representative geomagnetic-field magnitude to express MTQ dipole strength in
physical units; change ``B_REFERENCE_T`` for a mission-specific field model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


B_REFERENCE_T = 35e-6
N_SAMPLES = 400_000
RNG_SEED = 7
# 0.1 micro-N m to 1 mN m: low nanosatellite disturbances through demanding cases.
DISTURBANCE_LEVELS_MNM = (1e-4, 1e-3, 1e-2, 1e-1)
REJECTION_LEVEL_PERCENT = 90


def rejection_probability_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RW and MTQ normalized-authority axes and their coverage map."""
    rng = np.random.default_rng(RNG_SEED)
    wheel_axis = np.array([1.0, 0.0, 0.0])

    field = rng.normal(size=(N_SAMPLES, 3))
    field /= np.linalg.norm(field, axis=1)[:, None]
    disturbance = rng.normal(size=(N_SAMPLES, 3))
    disturbance /= np.linalg.norm(disturbance, axis=1)[:, None]

    axis_dot_field = field @ wheel_axis
    valid = np.abs(axis_dot_field) > 1e-6
    field = field[valid]
    disturbance = disturbance[valid]
    axis_dot_field = axis_dot_field[valid]

    wheel_torque = np.sum(disturbance * field, axis=1) / axis_dot_field
    rw_required = np.abs(wheel_torque)
    wheel_component = wheel_torque[:, None] * wheel_axis
    mtq_required = np.linalg.norm(disturbance - wheel_component, axis=1)

    eta_rw = np.logspace(-3, 5, 420)
    eta_mtq = np.logspace(-4, 6, 460)
    coverage = np.zeros((len(eta_mtq), len(eta_rw)))
    for index, authority in enumerate(eta_mtq):
        rw_subset = np.sort(rw_required[mtq_required <= authority])
        coverage[index, :] = np.searchsorted(rw_subset, eta_rw, side="right") / len(rw_required)

    return eta_rw, eta_mtq, coverage


def add_coverage_contours(
    ax: plt.Axes,
    eta_rw: np.ndarray,
    eta_mtq: np.ndarray,
    coverage: np.ndarray,
) -> None:
    """Draw the 90%-rejection boundary for each disturbance level."""
    colors = plt.get_cmap("plasma")(np.linspace(0.10, 0.90, len(DISTURBANCE_LEVELS_MNM)))

    for tau_disturbance_mnm, color in zip(DISTURBANCE_LEVELS_MNM, colors):
        rw_torque_mnm = eta_rw * tau_disturbance_mnm
        dipole_am2 = eta_mtq * (tau_disturbance_mnm * 1e-3) / B_REFERENCE_T
        ax.contour(
            rw_torque_mnm,
            dipole_am2,
            100 * coverage,
            levels=[REJECTION_LEVEL_PERCENT],
            colors=[color],
            linewidths=2.2,
            zorder=2,
        )


def add_heritage_points(ax: plt.Axes) -> None:
    """Add satellites, retaining incomplete public actuator data explicitly."""
    complete = [
        ("CanX-2", 3.0, 0.13, (4, 5), "left"),
        ("Ex-Alta 1", 15.0, 0.20, (4, 4), "left"),
        ("BRITE / GNB", 2.0, 0.12, (-28, -19), "right"),
        ("BeaverCube 2", 0.13, 0.20, (4, 4), "left"),
        ("ORCASat", 0.23, 0.13, (20, -19), "left"),
        ("NanoFF", 0.1, 0.30, (4, 5), "left"),
    ]
    rw_only = [
        ("SNAP-1\n(MTQ unreported)", 0.08),
        ("FASat-Bravo\n(MTQ unreported)", 5.0),
    ]
    mtq_only = [("Q-SAT\n(RW torque unreported)", 3.4)]

    for name, rw_torque_mnm, dipole_am2, offset, alignment in complete:
        ax.scatter(rw_torque_mnm, dipole_am2, s=36, marker="o", color="#ff7f0e", edgecolor="black", linewidth=0.4, zorder=4)
        ax.annotate(name, (rw_torque_mnm, dipole_am2), xytext=offset, textcoords="offset points", fontsize=5.5, ha=alignment)

    y_min = ax.get_ylim()[0]
    for name, rw_torque_mnm in rw_only:
        ax.scatter(rw_torque_mnm, y_min, s=40, marker="v", color="#e76f51", clip_on=False, zorder=4)
        ax.annotate(name, (rw_torque_mnm, y_min), xytext=(4, 5), textcoords="offset points", fontsize=5.5, va="bottom")

    x_min = ax.get_xlim()[0]
    for name, dipole_am2 in mtq_only:
        ax.scatter(x_min, dipole_am2, s=40, marker="<", color="#457b9d", clip_on=False, zorder=4)
        ax.annotate(name, (x_min, dipole_am2), xytext=(5, 3), textcoords="offset points", fontsize=5.5)


def main() -> Path:
    eta_rw, eta_mtq, coverage = rejection_probability_map()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 30)
    ax.set_ylim(3e-3, 3e2)

    add_coverage_contours(ax, eta_rw, eta_mtq, coverage)
    add_heritage_points(ax)

    disturbance_colors = plt.get_cmap("plasma")(np.linspace(0.10, 0.90, len(DISTURBANCE_LEVELS_MNM)))
    contour_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=rf"$\tau_d={tau:.4g}$ mN m")
        for tau, color in zip(DISTURBANCE_LEVELS_MNM, disturbance_colors)
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=5, label="Both actuator values reported"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e76f51", markersize=5, label="RW torque only"),
        Line2D([0], [0], marker="<", color="w", markerfacecolor="#457b9d", markersize=5, label="MTQ dipole only"),
    ]
    ax.legend(handles=contour_handles, loc="upper left", fontsize=4.8, ncol=2, framealpha=0.9)
    ax.grid(True, which="both", color="0.7", alpha=0.45)
    ax.set_xlabel(r"RW maximum torque $\tau_{\mathrm{RW,max}}$ [mN m]", fontsize=7)
    ax.set_ylabel(r"MTQ maximum dipole $D_{\max}$ [A m$^2$]", fontsize=7)
    ax.set_title("90% 3+1 disturbance-rejection trade-off", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.text(
        0.5,
        0.01,
        rf"MTQ torque authority evaluated at $\|\mathbf{{B}}\|={B_REFERENCE_T * 1e6:.0f}\,\mu$T",
        fontsize=5.5,
        ha="center",
    )

    output_path = Path(__file__).resolve().parent / "outputs" / "disturbance_rejection_physical_sizing_tradeoff.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=250)
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
