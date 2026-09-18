"""Plot physical RW/MTQ sizing trade-offs for 3+1 magnetic control.

The contours are geometry-only rejection probabilities.  They use a
representative geomagnetic-field magnitude to express MTQ dipole strength in
physical units; change ``B_REFERENCE_T`` for a mission-specific field model.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import configure_ieee_style


B_REFERENCE_T = 35e-6
N_SAMPLES = 400_000
RNG_SEED = 7
# 0.1 micro-N m to 1 mN m: low nanosatellite disturbances through demanding cases.
DISTURBANCE_LEVELS_MNM = (1e-4, 1e-3, 1e-2, 1e-1)
REJECTION_LEVEL_PERCENT = 90
MC_RW_TORQUE_MNM = 2.0
MC_MTQ_DIPOLE_AM2 = 0.6


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
    colors = plt.get_cmap("viridis")(
        np.linspace(0.10, 0.90, len(DISTURBANCE_LEVELS_MNM))
    )

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
        # The normalized RW grid ends at eta_rw=1e5.  For the smallest
        # disturbance this maps to 10 mN m, inside the displayed x-range;
        # continue the horizontal asymptote to the axis limit.
        target_coverage = REJECTION_LEVEL_PERCENT
        high_rw_coverage = coverage[:, -1]
        crossing = np.flatnonzero(high_rw_coverage >= target_coverage / 100.0)
        if crossing.size and rw_torque_mnm[-1] < ax.get_xlim()[1]:
            ax.plot(
                [rw_torque_mnm[-1], ax.get_xlim()[1]],
                [dipole_am2[crossing[0]], dipole_am2[crossing[0]]],
                color=color,
                linewidth=2.2,
                zorder=2,
            )


def add_heritage_points(ax: plt.Axes) -> None:
    """Add satellites, retaining incomplete public actuator data explicitly."""
    # Plain U labels are documented CubeSat form factors. Approximate labels
    # are volume equivalents (1 U ~= 1 litre) for non-CubeSat spacecraft.
    complete = [
        ("CanX-2 [3U]", 3.0, 0.13, (8, -5), "left"),
        ("Ex-Alta 1 [3U]", 15.0, 0.20, (4, 4), "left"),
        ("BRITE / GNB\n" r"[$\sim$8U equiv.]", 2.0, 0.12, (2, -20), "left"),
        ("BeaverCube 2 [3U]", 0.13, 0.20, (4, 4), "left"),
        ("ORCASat [2U]", 0.23, 0.13, (0, -18), "center"),
        ("NanoFF [2U]", 0.1, 0.30, (4, 5), "left"),
    ]
    rw_only = [
        ("SNAP-1\n" r"[$\sim$30U equiv.]", 0.08, (0, 5), "center"),
        ("FASat-Bravo\n" r"[$\sim$74U equiv.]", 5.0, (0, 5), "center"),
    ]
    mtq_only = [("Q-SAT\n" r"[$\sim$70U equiv.]", 3.4)]

    for name, rw_torque_mnm, dipole_am2, offset, alignment in complete:
        ax.scatter(rw_torque_mnm, dipole_am2, s=36, marker="o", color="#ff7f0e", edgecolor="black", linewidth=0.4, zorder=4)
        ax.annotate(
            name, (rw_torque_mnm, dipole_am2), xytext=offset,
            textcoords="offset points", fontsize=5.5, ha=alignment,
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.45,
                        "shrinkA": 2.0, "shrinkB": 2.0},
        )

    y_min = ax.get_ylim()[0]
    for name, rw_torque_mnm, offset, alignment in rw_only:
        ax.scatter(rw_torque_mnm, y_min, s=40, marker="v", color="#e76f51", clip_on=False, zorder=4)
        ax.annotate(
            name, (rw_torque_mnm, y_min), xytext=offset,
            textcoords="offset points", fontsize=5.5, va="bottom", ha=alignment,
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.45,
                        "shrinkA": 2.0, "shrinkB": 2.0},
        )

    x_min = ax.get_xlim()[0]
    for name, dipole_am2 in mtq_only:
        ax.scatter(x_min, dipole_am2, s=40, marker="<", color="#457b9d", clip_on=False, zorder=4)
        ax.annotate(
            name, (x_min, dipole_am2), xytext=(5, 3),
            textcoords="offset points", fontsize=5.5,
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.45,
                        "shrinkA": 2.0, "shrinkB": 2.0},
        )


def add_mc_configuration(ax: plt.Axes) -> None:
    """Mark the 6U actuator configuration used by the Monte Carlo campaigns."""
    ax.scatter(
        MC_RW_TORQUE_MNM,
        MC_MTQ_DIPOLE_AM2,
        s=72,
        marker="*",
        color="#C62828",
        edgecolor="black",
        linewidth=0.45,
        zorder=6,
    )
    ax.annotate(
        "6U MC configuration",
        (MC_RW_TORQUE_MNM, MC_MTQ_DIPOLE_AM2),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=5.5,
        color="#8B1A1A",
    )


def main() -> Path:
    configure_ieee_style()
    eta_rw, eta_mtq, coverage = rejection_probability_map()

    fig, ax = plt.subplots(figsize=(3.45, 3.25))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 30)
    ax.set_ylim(3e-3, 3e2)

    add_coverage_contours(ax, eta_rw, eta_mtq, coverage)
    add_heritage_points(ax)
    add_mc_configuration(ax)

    disturbance_colors = plt.get_cmap("viridis")(
        np.linspace(0.10, 0.90, len(DISTURBANCE_LEVELS_MNM))
    )
    contour_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=rf"$\tau_d={tau:.4g}$ mN m")
        for tau, color in zip(DISTURBANCE_LEVELS_MNM, disturbance_colors)
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=5, label="Both actuator values reported"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e76f51", markersize=5, label="RW torque reported only"),
        Line2D([0], [0], marker="<", color="w", markerfacecolor="#457b9d", markersize=5, label="MTQ dipole reported only"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#C62828",
               markeredgecolor="black", markersize=7, label="6U MC configuration"),
    ]
    ax.legend(handles=contour_handles, loc="upper left", fontsize=5.0,
              ncol=2, framealpha=0.92)
    ax.grid(True, which="both")
    ax.set_xlabel(r"RW maximum torque $\tau_{\mathrm{RW,max}}$ [mN m]")
    ax.set_ylabel(r"MTQ maximum dipole $D_{\max}$ [A m$^2$]")
    ax.set_title("90% 3+1 disturbance-rejection trade-off")
    ax.tick_params(which="both")
    fig.text(
        0.5,
        0.01,
        rf"MTQ torque authority evaluated at $\|\mathbf{{B}}\|={B_REFERENCE_T * 1e6:.0f}\,\mu$T",
        fontsize=5.5,
        ha="center",
    )

    output_path = Path(__file__).resolve().parent / "outputs" / "disturbance_rejection_physical_sizing_tradeoff.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
