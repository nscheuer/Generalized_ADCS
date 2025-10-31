"""
test_orbit_J2.py

Standalone visualization and test for RK4 orbit propagation with J2 perturbation.
Highlights the closest approach point to the initial position in color.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.orbital_state import Orbital_State


def closest_approach(R0, traj, min_skip=1):
    """Return index and distance of the closest approach to R0, skipping the first few samples."""
    d2 = np.sum((traj - R0) ** 2, axis=1)
    if min_skip > 0:
        d2[:min_skip] = np.inf
    i_star = np.argmin(d2)
    return i_star, np.sqrt(d2[i_star])


def run_orbit_rk4_J2(dt=60.0):
    """Propagate one orbit with RK4 and J2 perturbation."""
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
                          density_model=None, fast=False)

    T_orbit = 2 * np.pi * np.sqrt(r_mag ** 3 / mu)
    steps = int(T_orbit / dt)
    dt = T_orbit / steps
    times = np.linspace(0, T_orbit, steps + 1)

    positions = np.zeros((steps + 1, 3))
    positions[0, :] = orbit.R

    for i in tqdm(range(steps), desc="RK4 (J2=True)", ncols=80):
        orbit = orbit.propagate_orbit_rk4(dt, J2_perturbation_on=True)
        positions[i + 1, :] = orbit.R

    return times, positions


def main():
    times, positions = run_orbit_rk4_J2(dt=60.0)

    R0 = positions[0]
    i_star, d_min = closest_approach(R0, positions, min_skip=5)
    print(f"Closest approach at index {i_star} / {len(positions)-1}")
    print(f"Closest approach distance: {d_min:.6f} km")

    # === 2D X-Y Plot with color gradient ===
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = positions[:, 0], positions[:, 1]

    # Color map from start to finish
    sc = ax.scatter(x, y, c=np.arange(len(x)), cmap='viridis', s=8, label='Orbit path')
    ax.plot([R0[0]], [R0[1]], 'ro', label='Start point')
    ax.plot([x[i_star]], [y[i_star]], 'kx', markersize=10, label='Closest approach')

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_title("RK4 Orbit with J2 — Closest Approach Visualization")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Add colorbar for progression
    cbar = plt.colorbar(sc, ax=ax, label="Time step index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
