"""Compare TLE-fitted B* with geometry-averaged B* in SGP4Orbit."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import EarthSatellite

import ADCS
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants


TLE = (
    "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
    "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
)
MASS_KG = 10.0
DT_SECONDS = 300.0
PROPAGATION_HOURS = 12.0


def make_satellite() -> ADCS.Satellite:
    length, width, cd = 4.0, 0.2, 2.2
    faces = [
        ADCS.disturbances.GeometryFace(
            area=width**2,
            centroid=np.zeros(3),
            normal=np.array([sign, 0.0, 0.0]),
            CD=cd,
        )
        for sign in (-1.0, 1.0)
    ]
    faces.extend(
        ADCS.disturbances.GeometryFace(
            area=length * width,
            centroid=np.zeros(3),
            normal=normal,
            CD=cd,
        )
        for normal in (
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    drag = ADCS.disturbances.Drag_Disturbance(
        ADCS.disturbances.GeometryConfig(faces)
    )
    return ADCS.Satellite(
        mass=MASS_KG,
        J_0=np.diag([0.12, 13.35, 13.35]),
        disturbances=[drag],
    )


def main() -> None:
    environment = ADCS.Environment()
    satellite = make_satellite()
    source = EarthSatellite(*TLE, ts=environment.ephemeris.ts)
    start_time = float((source.epoch.tt - 2451545.0) / 36525.0)
    end_time = start_time + PROPAGATION_HOURS * 3600.0 * TimeConstants.sec2cent

    fitted_bstar = float(source.model.bstar)
    tle_orbit = ADCS.SGP4Orbit(
        TLE,
        end_time=end_time,
        dt=DT_SECONDS,
        bstar=fitted_bstar,  # Manual override; here we reuse the TLE's value.
        environment=environment,
    )
    geometry_orbit = ADCS.SGP4Orbit(
        TLE,
        end_time=end_time,
        dt=DT_SECONDS,
        satellite=satellite,
        environment=environment,
    )

    print(f"TLE fitted B*:       {fitted_bstar:.6e} 1/Earth radii")
    print(f"Geometry average B*: {geometry_orbit.bstar:.6e} 1/Earth radii")
    print(
        "Isotropic mean CD·A: "
        f"{geometry_orbit.mean_cd_area_m2:.3f} m²"
    )

    elapsed_hours = (tle_orbit.times - start_time) * TimeConstants.cent2sec / 3600.0
    tle_altitude = np.array([
        np.linalg.norm(tle_orbit.get_os(t).R) - EarthConstants.R_e
        for t in tle_orbit.times
    ])
    geometry_altitude = np.array([
        np.linalg.norm(geometry_orbit.get_os(t).R) - EarthConstants.R_e
        for t in geometry_orbit.times
    ])
    print(f"TLE B* altitude change:      {tle_altitude[-1] - tle_altitude[0]:.3f} km")
    print(
        "Geometry B* altitude change: "
        f"{geometry_altitude[-1] - geometry_altitude[0]:.3f} km"
    )

    fig, (ax, difference_ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax.plot(elapsed_hours, tle_altitude, label="TLE fitted B*")
    ax.plot(elapsed_hours, geometry_altitude, label="Geometry averaged B*")
    ax.set_title("SGP4 propagation with fitted and geometry averaged B*")
    ax.set_xlabel("Elapsed time [hours]")
    ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    difference_ax.plot(
        elapsed_hours,
        (geometry_altitude - tle_altitude) * 1000.0,
        color="tab:purple",
    )
    difference_ax.axhline(0.0, color="black", linewidth=0.8)
    difference_ax.set_xlabel("Elapsed time [hours]")
    difference_ax.set_ylabel("Difference [m]")
    difference_ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
