import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants


# Propagation setup
DT_SECONDS = 20.0
TF_SECONDS = 6.0 * 3600.0
ZONAL_CHOICES = [0, 2, 3, 4, 5, 6]

# A mildly eccentric, inclined orbit makes odd/even zonals easier to see.
R0_KM = np.array([6878.0, 0.0, 0.0])
V0_KMPS = np.array([0.0, 6.85, 3.15])
J2000_START = 0.22


def zonal_label(zonal_J: int) -> str:
    if zonal_J == 0:
        return "Two-body"
    if zonal_J == 2:
        return "J2"
    return f"J2-J{zonal_J}"


def rk4_step(
    R: np.ndarray,
    V: np.ndarray,
    dt: float,
    *,
    mu_e: float,
    R_e: float,
    J2coeff: float,
    Jcoeffs: np.ndarray,
    zonal_J: int,
) -> tuple[np.ndarray, np.ndarray]:
    k1r, k1v = Orbital_State._orbit_dynamics_raw(
        R, V, mu_e, R_e, J2coeff, zonal_J, Jcoeffs=Jcoeffs
    )
    k2r, k2v = Orbital_State._orbit_dynamics_raw(
        R + 0.5 * dt * k1r,
        V + 0.5 * dt * k1v,
        mu_e,
        R_e,
        J2coeff,
        zonal_J,
        Jcoeffs=Jcoeffs,
    )
    k3r, k3v = Orbital_State._orbit_dynamics_raw(
        R + 0.5 * dt * k2r,
        V + 0.5 * dt * k2v,
        mu_e,
        R_e,
        J2coeff,
        zonal_J,
        Jcoeffs=Jcoeffs,
    )
    k4r, k4v = Orbital_State._orbit_dynamics_raw(
        R + dt * k3r,
        V + dt * k3v,
        mu_e,
        R_e,
        J2coeff,
        zonal_J,
        Jcoeffs=Jcoeffs,
    )

    R_next = R + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
    V_next = V + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return R_next, V_next


def propagate_histories(
    state0: Orbital_State,
    zonal_choices: list[int],
    tf_seconds: float,
    dt_seconds: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    steps = int(round(tf_seconds / dt_seconds))
    times = np.linspace(0.0, steps * dt_seconds, steps + 1)

    mu_e = float(state0.mu_e)
    R_e = float(state0.R_e)
    J2coeff = float(state0.J2coeff)
    Jcoeffs = np.asarray(state0.Jcoeffs, dtype=float)

    pos_hist: dict[int, np.ndarray] = {}
    vel_hist: dict[int, np.ndarray] = {}

    for zonal_J in zonal_choices:
        R = np.asarray(state0.R, dtype=float).copy()
        V = np.asarray(state0.V, dtype=float).copy()

        pos = np.empty((steps + 1, 3), dtype=float)
        vel = np.empty((steps + 1, 3), dtype=float)
        pos[0, :] = R
        vel[0, :] = V

        for k in range(steps):
            R, V = rk4_step(
                R,
                V,
                dt_seconds,
                mu_e=mu_e,
                R_e=R_e,
                J2coeff=J2coeff,
                Jcoeffs=Jcoeffs,
                zonal_J=zonal_J,
            )
            pos[k + 1, :] = R
            vel[k + 1, :] = V

        pos_hist[zonal_J] = pos
        vel_hist[zonal_J] = vel

    return times, pos_hist, vel_hist


def decompose_rtn(delta_r: np.ndarray, ref_r: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    rtn = np.empty_like(delta_r)
    for k in range(delta_r.shape[0]):
        rhat = ref_r[k] / np.linalg.norm(ref_r[k])
        hhat = np.cross(ref_r[k], ref_v[k])
        hhat = hhat / np.linalg.norm(hhat)
        that = np.cross(hhat, rhat)
        basis = np.column_stack([rhat, that, hhat])
        rtn[k, :] = basis.T @ delta_r[k]
    return rtn


def add_earth(ax) -> None:
    radius = EarthConstants.R_e
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, np.pi, 40)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#7fb3d5", alpha=0.18, linewidth=0.0, zorder=0)


def plot_zonal_comparison(times, pos_hist, vel_hist, zonal_choices) -> None:
    colors = cm.viridis(np.linspace(0.05, 0.95, len(zonal_choices)))
    t_hours = times / 3600.0

    ref0 = pos_hist[0]
    ref2 = pos_hist[2]
    ref2_vel = vel_hist[2]

    fig = plt.figure(figsize=(16, 9))
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    add_earth(ax3d)

    for color, zonal_J in zip(colors, zonal_choices):
        pos = pos_hist[zonal_J]
        ax3d.plot(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            color=color,
            linewidth=1.8,
            label=zonal_label(zonal_J),
        )
        ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, s=18)

    ax3d.set_title("3D Orbit Propagation With Different Zonal Harmonics")
    ax3d.set_xlabel("ECI X [km]")
    ax3d.set_ylabel("ECI Y [km]")
    ax3d.set_zlabel("ECI Z [km]")
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.legend(loc="upper left", fontsize=9)

    ax_xy = fig.add_subplot(2, 3, 2)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_xy.plot(
        EarthConstants.R_e * np.cos(theta),
        EarthConstants.R_e * np.sin(theta),
        color="0.7",
        linestyle="--",
        linewidth=1.0,
        label="Earth radius",
    )
    for color, zonal_J in zip(colors, zonal_choices):
        pos = pos_hist[zonal_J]
        ax_xy.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5, label=zonal_label(zonal_J))
    ax_xy.set_title("Equatorial Projection")
    ax_xy.set_xlabel("ECI X [km]")
    ax_xy.set_ylabel("ECI Y [km]")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="box")

    ax_sep = fig.add_subplot(2, 3, 3)
    for color, zonal_J in zip(colors[1:], zonal_choices[1:]):
        sep = np.linalg.norm(pos_hist[zonal_J] - ref0, axis=1)
        ax_sep.plot(t_hours, sep, color=color, linewidth=1.8, label=f"{zonal_label(zonal_J)} - Two-body")
    ax_sep.set_title("Position Separation From Two-Body Reference")
    ax_sep.set_xlabel("Time [hours]")
    ax_sep.set_ylabel(r"$\|\Delta r\|$ [km]")
    ax_sep.grid(True, alpha=0.3)
    ax_sep.legend(fontsize=8)

    ax_rtn = fig.add_subplot(2, 3, 5)
    for zonal_J, style, comp_idx in [(3, "--", 2), (4, "-.", 2), (5, ":", 2), (6, "-", 2)]:
        delta = pos_hist[zonal_J] - ref2
        rtn = decompose_rtn(delta, ref2, ref2_vel)
        ax_rtn.plot(
            t_hours,
            rtn[:, comp_idx],
            linestyle=style,
            linewidth=1.6,
            label=f"{zonal_label(zonal_J)} - J2",
        )
    ax_rtn.set_title("Cross-Track Shift Relative to J2")
    ax_rtn.set_xlabel("Time [hours]")
    ax_rtn.set_ylabel("Normal RTN component [km]")
    ax_rtn.grid(True, alpha=0.3)
    ax_rtn.legend(fontsize=8)

    ax_alt = fig.add_subplot(2, 3, 6)
    for color, zonal_J in zip(colors, zonal_choices):
        altitude = np.linalg.norm(pos_hist[zonal_J], axis=1) - EarthConstants.R_e
        ax_alt.plot(t_hours, altitude, color=color, linewidth=1.8, label=zonal_label(zonal_J))
    ax_alt.set_title("Altitude History")
    ax_alt.set_xlabel("Time [hours]")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.grid(True, alpha=0.3)

    fig.tight_layout()


def print_summary(times, pos_hist) -> None:
    t_hours = times / 3600.0
    ref0 = pos_hist[0]
    ref2 = pos_hist[2]
    print("Peak position separation over propagation window")
    print("-" * 52)
    for zonal_J in ZONAL_CHOICES[1:]:
        sep0 = np.linalg.norm(pos_hist[zonal_J] - ref0, axis=1)
        sep2 = np.linalg.norm(pos_hist[zonal_J] - ref2, axis=1)
        i0 = int(np.argmax(sep0))
        i2 = int(np.argmax(sep2))
        print(
            f"{zonal_label(zonal_J):>8s} : "
            f"vs two-body = {sep0[i0]:9.4f} km at {t_hours[i0]:5.2f} h, "
            f"vs J2 = {sep2[i2]:9.4f} km at {t_hours[i2]:5.2f} h"
        )


def main() -> None:
    ephem = Ephemeris()
    state0 = Orbital_State(
        ephem=ephem,
        J2000=J2000_START,
        R=R0_KM,
        V=V0_KMPS,
    )

    times, pos_hist, vel_hist = propagate_histories(
        state0,
        zonal_choices=ZONAL_CHOICES,
        tf_seconds=TF_SECONDS,
        dt_seconds=DT_SECONDS,
    )

    print_summary(times, pos_hist)
    plot_zonal_comparison(times, pos_hist, vel_hist, ZONAL_CHOICES)
    plt.show()


if __name__ == "__main__":
    main()
