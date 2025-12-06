import sys
import os
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize


def closest_approach(R0, traj, min_skip=1):
    """Return index and distance of the closest approach to R0.
    Skips the first `min_skip` samples to avoid matching the same point."""
    d2 = np.sum((traj - R0) ** 2, axis=1)
    if min_skip > 0:
        d2[:min_skip] = np.inf
    i_star = np.argmin(d2)
    return i_star, np.sqrt(d2[i_star])


def test_orbit(method="rk4", use_J2=False, dt=60.0):
    ephem = Ephemeris()
    mu = EarthConstants.mu_e
    Re = EarthConstants.R_e

    alt = 400.0  # km
    r_mag = Re + alt
    v_mag = np.sqrt(mu / r_mag)

    R0 = np.array([r_mag, 0, 0])
    V0 = np.array([0, v_mag, 0])

    orbit = Orbital_State(ephem, J2000=0.0, R=R0, V=V0,
                          S=None, B=None, rho=None,
                          density_model=None, fast=True)

    # === Orbital period and adjusted dt ===
    T_orbit = 2 * np.pi * np.sqrt(r_mag ** 3 / mu)
    steps = int(T_orbit / dt)
    dt = T_orbit / steps  # adjust so exactly one full orbit
    times = np.linspace(0, T_orbit, steps + 1)

    positions = np.zeros((steps + 1, 3))
    positions[0, :] = orbit.R  # initial state

    # === Propagate ===
    for i in tqdm(range(steps), desc=f"{method.upper()} (J2={use_J2})", ncols=80):
        if method == "rk4":
            orbit = orbit.propagate_orbit_rk4(dt, use_J2)
        elif method == "euler":
            orbit = orbit.propagate_orbit(dt, use_J2)
        else:
            raise ValueError(f"Unknown method: {method}")
        positions[i + 1, :] = orbit.R

    return times, positions


def test_orbit_dyn_Jacobians():
    for k in range(1):
        pos = 7000*random_n_unit_vec(3)
        vel = 8*normalize(np.cross(random_n_unit_vec(3), pos))
        ephem = Ephemeris()
        os = Orbital_State(ephem=ephem, J2000=0.22, R=pos, V=vel)

        rfun = lambda c: Orbital_State(ephem=ephem, J2000=0.22, R=np.array([c[0],c[1],c[2]]), V=np.array([c[3],c[4],c[5]]), fast=True).orbit_dynamics(J2_perturbation_on=True)[0]
        vfun = lambda c: Orbital_State(ephem=ephem, J2000=0.22, R=np.array([c[0],c[1],c[2]]), V=np.array([c[3],c[4],c[5]]), fast=True).orbit_dynamics(J2_perturbation_on=True)[1]

        Jrfun = nd.Jacobian(rfun)(pos.flatten().tolist() + vel.flatten().tolist())
        Jvfun = nd.Jacobian(vfun)(pos.flatten().tolist() + vel.flatten().tolist())
        drd_dr, drd_dv, dvd_dr, dvd_dv = os.orbit_dynamics_jacobians(J2_perturbation_on=True)

        # Build full 6×6 analytic Jacobian
        combined_results = np.block([
            [drd_dr, dvd_dr],
            [drd_dv, dvd_dv],
        ])

        Jrfuntest = np.array(Jrfun)
        Jvfuntest = np.array(Jvfun)
        assert np.allclose(Jrfuntest.T,combined_results[:,0:3])
        assert np.allclose(Jvfuntest.T,combined_results[:,3:6])

def test_orbit_rk4_Jacobians():
    for k in range(1):
        pos = 7000*random_n_unit_vec(3)
        vel = 8*normalize(np.cross(random_n_unit_vec(3), pos))
        ephem = Ephemeris()
        os = Orbital_State(ephem=ephem, J2000=0.22, R=pos, V=vel)
        dt = 1.0

        rfun = lambda c: Orbital_State(ephem=ephem, J2000=0.22, R=np.array([c[0],c[1],c[2]]), V=np.array([c[3],c[4],c[5]]), fast=True).propagate_orbit_rk4(dt=dt, J2_perturbation_on=True).R
        vfun = lambda c: Orbital_State(ephem=ephem, J2000=0.22, R=np.array([c[0],c[1],c[2]]), V=np.array([c[3],c[4],c[5]]), fast=True).propagate_orbit_rk4(dt=dt, J2_perturbation_on=True).V

        Jrfun = nd.Jacobian(rfun)(pos.flatten().tolist() + vel.flatten().tolist())
        Jvfun = nd.Jacobian(vfun)(pos.flatten().tolist() + vel.flatten().tolist())
        [drd__dr,drd__dv,dvd__dr,dvd__dv] = os.propagate_jacobians_rk4(dt=dt, J2_perturbation_on=True)

        Jrfuntest = np.array(Jrfun)
        Jvfuntest = np.array(Jvfun)

        assert np.allclose(Jrfuntest.T, np.vstack([drd__dr, drd__dv]))
        assert np.allclose(Jvfuntest.T, np.vstack([dvd__dr, dvd__dv]))




def main():
    """Run visual orbit propagation comparisons (for human run)."""
    results = {}

    # === Run all four simulations once ===
    for method in ["euler", "rk4"]:
        for use_J2 in [False, True]:
            key = f"{method}_{'J2' if use_J2 else 'noJ2'}"
            times, positions = test_orbit(method=method, use_J2=use_J2, dt=120.0)
            results[key] = {"times": times, "positions": positions}

    # === 1. Altitude vs Time Plot ===
    plt.figure(figsize=(10, 6))
    Re = EarthConstants.R_e

    for key, data in results.items():
        altitudes = np.linalg.norm(data["positions"], axis=1) - Re
        label = key.replace("_", " ").upper()
        plt.plot(data["times"] / 3600, altitudes, label=label,
                 linestyle='--' if 'euler' in key else '-')

    plt.xlabel('Time [hours]')
    plt.ylabel('Altitude [km]')
    plt.title('Orbit Propagation Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # === 2. 2D X-Y Orbit Plot ===
    fig_xy = plt.figure(figsize=(8, 8))
    ax_xy = fig_xy.add_subplot(111)
    for key, data in results.items():
        pos = data["positions"]
        label = key.replace("_", " ").upper()
        ax_xy.plot(pos[:, 0], pos[:, 1],
                   '--' if 'euler' in key else '-', label=label)
    ax_xy.set_xlabel('X [km]')
    ax_xy.set_ylabel('Y [km]')
    ax_xy.set_title('2D Orbit Projection (X–Y Plane)')
    ax_xy.legend()
    ax_xy.grid(True)

    all_xy = np.vstack([res["positions"][:, :2] for res in results.values()])
    max_range = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) / 2
    mid_x = np.mean([np.max(all_xy[:, 0]), np.min(all_xy[:, 0])])
    mid_y = np.mean([np.max(all_xy[:, 1]), np.min(all_xy[:, 1])])
    ax_xy.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_xy.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_xy.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    plt.close()

    # === 3. 3D Orbit Plot ===
    fig_3d = plt.figure(figsize=(9, 9))
    ax3d = fig_3d.add_subplot(111, projection='3d')
    for key, data in results.items():
        pos = data["positions"]
        label = key.replace("_", " ").upper()
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                  '--' if 'euler' in key else '-', label=label)
    ax3d.set_xlabel('X [km]')
    ax3d.set_ylabel('Y [km]')
    ax3d.set_zlabel('Z [km]')
    ax3d.set_title('3D Orbit Trajectories')
    ax3d.legend()

    all_coords = np.vstack([res["positions"] for res in results.values()])
    max_range = np.ptp(all_coords, axis=0).max() / 2.0
    mid_x = (np.max(all_coords[:, 0]) + np.min(all_coords[:, 0])) / 2.0
    mid_y = (np.max(all_coords[:, 1]) + np.min(all_coords[:, 1])) / 2.0
    mid_z = (np.max(all_coords[:, 2]) + np.min(all_coords[:, 2])) / 2.0
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.show()

    # === Improved orbit closure check ===
    start_rk4_no_J2 = results["rk4_noJ2"]["positions"][0]
    _, min_err_noJ2 = closest_approach(start_rk4_no_J2, results["rk4_noJ2"]["positions"])
    print(f"[RK4 no J2] Closest approach distance: {min_err_noJ2:.6f} km")

    start_rk4_J2 = results["rk4_J2"]["positions"][0]
    _, min_err_J2 = closest_approach(start_rk4_J2, results["rk4_J2"]["positions"])
    print(f"[RK4 J2] Closest approach distance: {min_err_J2:.6f} km")


@pytest.mark.parametrize("use_J2", [False])
def test_orbit_closes(use_J2):
    """
    Test that the RK4 orbit returns within 1 km of its original track after one orbit.
    For J2, this uses closest approach instead of exact closure.
    """
    _, positions = test_orbit(method="rk4", use_J2=use_J2, dt=60.0)
    i_star, d_min = closest_approach(positions[0], positions)
    print(f"[RK4 J2={use_J2}] Closest approach distance: {d_min:.6f} km")
    assert d_min < 1.0, f"Orbit track drifted too far: {d_min:.3f} km"


if __name__ == "__main__":
    main()
