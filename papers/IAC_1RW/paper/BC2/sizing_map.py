"""First-principles 3+1 MTQ and reaction-wheel sizing screens."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_style import configure_ieee_style


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
R_E_KM = 6378.137
MU_KM3_S2 = 398600.4418
MU_M3_S2 = MU_KM3_S2 * 1.0e9
B_400_T = 35.0e-6
RHO_400 = 3.9e-12
RHO_SCALE_HEIGHT_KM = 60.0

# BC2 factory hardware values.
M_MAX_BC2_AM2 = 0.20
H_MAX_BC2_NMS = 3.6e-3
# CanX-2 reference values: mission-reported 10 x 10 x 34 cm geometry, 635 km,
# and 30 mN m s wheel storage; 0.1 A m² is a literature coil reference value.
M_MAX_CANX2_AM2 = 0.10
H_MAX_CANX2_NMS = 30.0e-3
REFERENCE_ALTITUDE_KM = 635.0
CANX2_GEOMETRY_SCALE = (0.34 / 0.30) ** (1.0 / 3.0)
CONTACT_STORAGE_WINDOW_S = 600.0


def magnetic_field_t(altitude_km: np.ndarray) -> np.ndarray:
    """Dipole field magnitude anchored at 35 microtesla at 400 km."""
    return B_400_T * ((R_E_KM + 400.0) / (R_E_KM + altitude_km)) ** 3


def disturbance_torque_nm(altitude_km: np.ndarray, bus_scale: np.ndarray) -> np.ndarray:
    """Conservative sum of gravity-gradient, drag, SRP, and residual-dipole torque.

    The generic reference is a 0.1 x 0.1 x 0.3 m 3U bus.  Uniform geometric
    scaling gives inertia difference proportional to s^5, projected area to
    s^2, lever arm to s, and residual dipole to s^3.
    """
    radius_m = (R_E_KM + altitude_km) * 1.0e3
    velocity_m_s = np.sqrt(MU_M3_S2 / radius_m)
    density = RHO_400 * np.exp(-(altitude_km - 400.0) / RHO_SCALE_HEIGHT_KM)
    gg = 3.0 * MU_M3_S2 / radius_m**3 * (0.02 * bus_scale**5)
    drag = 0.5 * density * velocity_m_s**2 * 2.2 * (0.03 * bus_scale**2) * (0.05 * bus_scale)
    srp = 4.56e-6 * 1.3 * (0.03 * bus_scale**2) * (0.05 * bus_scale)
    residual_dipole = (0.01 * bus_scale**3) * magnetic_field_t(altitude_km)
    return gg + drag + srp + residual_dipole


def required_mtq_dipole_am2(altitude_km: np.ndarray, bus_scale: np.ndarray) -> np.ndarray:
    """Orbit-averaged MTQ dipole required for continuous torque rejection."""
    rms_transverse_factor = np.sqrt(2.0 / 3.0)
    return disturbance_torque_nm(altitude_km, bus_scale) / (
        rms_transverse_factor * magnetic_field_t(altitude_km)
    )


def required_wheel_storage_nms(altitude_km: np.ndarray, bus_scale: np.ndarray) -> np.ndarray:
    """Conservative wheel storage for a 10 min contact with no unloading.

    This assumes the full disturbance torque is stored by the wheel.  It is a
    deliberately conservative capacity screen, not a desaturation model; the
    latter belongs in the momentum-management/science-uptime analysis.
    """
    return disturbance_torque_nm(altitude_km, bus_scale) * CONTACT_STORAGE_WINDOW_S


def _add_spacecraft_markers(ax: plt.Axes) -> None:
    ax.scatter(REFERENCE_ALTITUDE_KM, 1.0, marker="*", s=76, color="#0072B2",
               edgecolor="black", linewidth=0.4, zorder=4)
    ax.annotate("BeaverCube II", (REFERENCE_ALTITUDE_KM, 1.0), xytext=(-54, -15),
                textcoords="offset points", fontsize=5.7, color="#0072B2")
    ax.scatter(REFERENCE_ALTITUDE_KM, CANX2_GEOMETRY_SCALE, marker="P", s=38,
               color="#CC79A7", edgecolor="black", linewidth=0.35, zorder=4)
    ax.annotate("CanX-2", (REFERENCE_ALTITUDE_KM, CANX2_GEOMETRY_SCALE), xytext=(5, 7),
                textcoords="offset points", fontsize=5.7, color="#A23B72")


def _format_map_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Linear bus / geometry scale\n(relative to generic 3U)")
    ax.set_title(title)


def main() -> None:
    configure_ieee_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use the longest bus dimension as the design variable.  The model keeps
    # the 3U aspect ratio fixed, so L = 0.30 m corresponds to scale = 1.
    bus_scale = np.linspace(0.5, 4.0, 241)
    long_dimension_m = 0.30 * bus_scale
    altitude_lines_km = (300.0, 400.0, 500.0, 635.0, 800.0, 1000.0, 1200.0)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(altitude_lines_km)))

    fig, (ax_mtq, ax_rw) = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)
    mtq_handles = []
    rw_handles = []
    for altitude_km, color in zip(altitude_lines_km, colors):
        dipole = required_mtq_dipole_am2(altitude_km, bus_scale)
        storage = required_wheel_storage_nms(altitude_km, bus_scale) * 1.0e3
        mtq_handles.append(ax_mtq.plot(long_dimension_m, dipole, color=color,
                                       linewidth=1.25, label=f"{altitude_km} km")[0])
        rw_handles.append(ax_rw.plot(long_dimension_m, storage, color=color,
                                     linewidth=1.25, label=f"{altitude_km} km")[0])

    # Hardware limits are horizontal because the question is whether a chosen
    # actuator can support a selected bus size at a given orbit altitude.
    ax_mtq.axhline(M_MAX_BC2_AM2, color="black", linewidth=1.0, label="BC2 limit")
    ax_mtq.axhline(M_MAX_CANX2_AM2, color="#555555", linestyle="--", linewidth=0.9,
                   label="CanX-2 reference")
    ax_rw.axhline(H_MAX_BC2_NMS * 1.0e3, color="black", linewidth=1.0, label="BC2 limit")
    ax_rw.axhline(H_MAX_CANX2_NMS * 1.0e3, color="#555555", linestyle="--", linewidth=0.9,
                  label="CanX-2 reference")

    # Mark the two spacecraft at the 635 km reference line.  The markers show
    # their physical size, while the horizontal lines show actuator capability.
    bc2_req_m = required_mtq_dipole_am2(REFERENCE_ALTITUDE_KM, np.array(1.0))
    canx_req_m = required_mtq_dipole_am2(REFERENCE_ALTITUDE_KM, np.array(CANX2_GEOMETRY_SCALE))
    bc2_req_h = required_wheel_storage_nms(REFERENCE_ALTITUDE_KM, np.array(1.0)) * 1.0e3
    canx_req_h = required_wheel_storage_nms(REFERENCE_ALTITUDE_KM, np.array(CANX2_GEOMETRY_SCALE)) * 1.0e3
    for ax, y_bc2, y_canx in ((ax_mtq, bc2_req_m, canx_req_m), (ax_rw, bc2_req_h, canx_req_h)):
        ax.scatter(0.30, y_bc2, marker="*", s=58, color="#0072B2", edgecolor="black", linewidth=0.35, zorder=4)
        ax.scatter(0.34, y_canx, marker="P", s=30, color="#CC79A7", edgecolor="black", linewidth=0.35, zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(0.15, 1.20)
        ax.grid(True, which="both", alpha=0.55)

    ax_mtq.set_xlabel("Satellite long dimension $L$ [m]")
    ax_mtq.set_ylabel("Required MTQ dipole $m_{req}$ [A m$^2$]")
    ax_mtq.set_title("A. MTQ requirement")
    ax_rw.set_xlabel("Satellite long dimension $L$ [m]")
    ax_rw.set_ylabel("Required RW storage $H_{req}$ [mN m s]")
    ax_rw.set_title("B. RW momentum requirement")
    ax_mtq.legend(handles=mtq_handles, title="Altitude", fontsize=5.2, title_fontsize=5.5, loc="upper left", ncol=2)
    ax_rw.legend(handles=rw_handles, title="Altitude", fontsize=5.2, title_fontsize=5.5, loc="upper left", ncol=2)

    fig.text(
        0.5, -0.025,
        "Solid/dashed horizontal lines: BC2 / CanX-2 actuator capability.  "
        "Markers: BC2 star, CanX-2 pentagon at 635 km.  "
        "Fixed 3U aspect-ratio scaling; RW panel assumes 10 min with no unloading.",
        ha="center", va="top", fontsize=5.8,
    )
    output = OUTPUT_DIR / "bc2_mtq_and_rw_sizing_map.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
