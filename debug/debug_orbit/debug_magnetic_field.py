import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List, Union
from tqdm import tqdm
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants

if __name__ == "__main__":
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]))
    orbit = Orbit(os0, end_time=0.22 + 2000*TimeConstants.sec2cent, dt=10, zonal_J=0, fast=False)

    times = sorted(orbit.states.keys())

    R_list = []
    B_list = []

    for t in times:
        state = orbit.states[t]

        # ensure 1D numpy vectors
        R_list.append(np.asarray(state.R).reshape(3))
        B_list.append(np.asarray(state.B).reshape(3))

    R = np.stack(R_list, axis=1)   # shape (3, N)
    B = np.stack(B_list, axis=1)
    print("First B:", B[:, 0])
    print("Last  B:", B[:, -1])

    N = R.shape[1]
    Bmag = np.linalg.norm(B, axis=0)

    plt.figure()
    plt.plot(Bmag)
    plt.title("Magnetic field magnitude along orbit")
    plt.xlabel("step")
    plt.ylabel("|B| (T)")
    plt.grid(True)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(R[0], R[1], R[2], label="orbit")

    # draw Earth
    RE = 6371  # km if your R is km
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = RE * np.outer(np.cos(u), np.sin(v))
    y = RE * np.outer(np.sin(u), np.sin(v))
    z = RE * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.2)

    # downsample B vectors
    step = max(1, N // 40)
    scale = 0.3 * np.max(np.linalg.norm(R, axis=0))

    ax.quiver(
        R[0, ::step],
        R[1, ::step],
        R[2, ::step],
        B[0, ::step],
        B[1, ::step],
        B[2, ::step],
        length=scale,
        normalize=True
    )

    ax.set_title("Orbit with magnetic field vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    fig2, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    axs[0].plot(B[0])
    axs[0].set_ylabel("Bx (T)")
    axs[0].grid(True)

    axs[1].plot(B[1])
    axs[1].set_ylabel("By (T)")
    axs[1].grid(True)

    axs[2].plot(B[2])
    axs[2].set_ylabel("Bz (T)")
    axs[2].set_xlabel("step")
    axs[2].grid(True)

    fig2.suptitle("Magnetic field components (ECI)")

    plt.show()
