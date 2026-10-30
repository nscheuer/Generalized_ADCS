"""Demonstrate attitude-coupled drag decay for a very low Earth orbit."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.orbits.universal_constants import EarthConstants


J2000_START = 0.22
INITIAL_ALTITUDE_KM = 250.0
SATELLITE_MASS_KG = 4.0
DRAG_AREA_M2 = 0.5
DRAG_COEFFICIENT = 2.2
DT_SECONDS = 60.0
SIMULATION_HOURS = 4.0


def main() -> None:
    environment = ADCS.Environment(atmosphere=ADCS.SMADAtmosphere())
    radius = EarthConstants.R_e + INITIAL_ALTITUDE_KM
    position = np.array([radius, 0.0, 0.0])
    velocity = np.array([0.0, np.sqrt(EarthConstants.mu_e / radius), 0.0])
    os0 = ADCS.Orbital_State.from_position(
        J2000=J2000_START,
        R=position,
        V=velocity,
        environment=environment,
    )

    # The +Y face points into the ram flow for this initial attitude and
    # circular-orbit direction. A centered face adds drag force without a
    # drag torque, keeping this example focused on orbital decay.
    drag = ADCS.disturbances.Drag_Disturbance(
        ADCS.disturbances.GeometryConfig([
            ADCS.disturbances.GeometryFace(
                area=DRAG_AREA_M2,
                centroid=np.zeros(3),
                normal=np.array([0.0, 1.0, 0.0]),
                CD=DRAG_COEFFICIENT,
            )
        ])
    )
    satellite = ADCS.Satellite(
        mass=SATELLITE_MASS_KG,
        J_0=np.diag([0.08, 0.08, 0.05]),
        disturbances=[drag],
    )
    x0 = ADCS.State.from_array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    orbit = ADCS.RK4Orbit(os0, environment=environment, zonal_J=2)
    results = ADCS.simulate(
        x=x0,
        satellite=satellite,
        orbit=orbit,
        environment=environment,
        dt=DT_SECONDS,
        tf=SIMULATION_HOURS * 3600.0,
    )

    times_hours = np.asarray(orbit.times - orbit.times[0]) * 36525.0 * 24.0
    altitude_km = np.array([
        np.linalg.norm(orbit.get_os(t).R) - EarthConstants.R_e for t in orbit.times
    ])
    print(f"Initial altitude: {altitude_km[0]:.2f} km")
    print(f"Final altitude:   {altitude_km[-1]:.2f} km")
    print(f"Altitude change:  {altitude_km[-1] - altitude_km[0]:.2f} km")
    print(f"Logged attitude states: {len(results.first().state_hist)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_hours, altitude_km, color="tab:blue", linewidth=2)
    ax.set_title("VLEO orbital decay with attitude-coupled drag")
    ax.set_xlabel("Elapsed time [hours]")
    ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
