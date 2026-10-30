"""Compare drag-induced orbital decay from SGP4 and RK4Orbit.

SGP4 and TLE conventions follow Vallado et al.,
`AIAA-2006-6753 <https://celestrak.org/publications/AIAA/2006-6753/>`_ and the
`CelesTrak TLE FAQ <https://celestrak.org/columns/v04n03/>`_. The comparison's
NRLMSISE-00 density model follows Picone et al. (2002); see the package citation
guide for its DOI and the reference for the NRLMSISE-00 implementation.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ADCS
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants


# ISS-like TLE at a very low mean altitude (about 250 km), kept fixed so the
# example is reproducible and does not depend on downloading current elements.
TLE = (
    "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
    "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 16.10000000 21106",
)
MASS_KG = 10.0
BODY_LENGTH_M = 4.0
BODY_WIDTH_M = 0.2
DRAG_COEFFICIENT = 2.2
DT_SECONDS = 60.0
PROPAGATION_HOURS = 6.0
# Diagnostic calibration for this specific epoch, atmosphere, geometry, and
# six-hour comparison. It is not a transferable physical B* estimate.
CALIBRATED_BSTAR = 9.5e-3


def make_satellite(*, include_drag: bool) -> ADCS.Satellite:
    disturbances = []
    if include_drag:
        end_area = BODY_WIDTH_M**2
        side_area = BODY_LENGTH_M * BODY_WIDTH_M
        faces = [
            ADCS.disturbances.GeometryFace(
                area=end_area,
                centroid=np.zeros(3),
                normal=np.array([sign, 0.0, 0.0]),
                CD=DRAG_COEFFICIENT,
            )
            for sign in (-1.0, 1.0)
        ]
        faces.extend(
            ADCS.disturbances.GeometryFace(
                area=side_area,
                centroid=np.zeros(3),
                normal=normal,
                CD=DRAG_COEFFICIENT,
            )
            for normal in (
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
                np.array([0.0, 0.0, 1.0]),
            )
        )
        disturbances.append(
            ADCS.disturbances.Drag_Disturbance(
                ADCS.disturbances.GeometryConfig(faces)
            )
        )
    return ADCS.Satellite(
        mass=MASS_KG,
        J_0=np.diag([0.12, 13.35, 13.35]),
        disturbances=disturbances,
    )


def semimajor_axis(os: ADCS.Orbital_State) -> float:
    radius = np.linalg.norm(os.R)
    specific_energy = 0.5 * np.dot(os.V, os.V) - EarthConstants.mu_e / radius
    return -EarthConstants.mu_e / (2.0 * specific_energy)


def propagate_rk4(
    initial: ADCS.Orbital_State,
    satellite: ADCS.Satellite,
    *,
    steps: int,
) -> list[float]:
    orbit = ADCS.RK4Orbit(initial, environment=initial.environment, zonal_J=2)
    # Keep the body fixed in ECI to make this a pure geometry/force comparison.
    fixed_attitude = ADCS.State.from_array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    for _ in tqdm(range(steps), desc="RK4Orbit", unit="step"):
        orbit.advance(fixed_attitude, fixed_attitude, satellite, DT_SECONDS)
    return [semimajor_axis(orbit.get_os(float(t))) for t in orbit.times]


def main() -> None:
    # Fixed solar and geomagnetic indices make this standalone comparison
    # reproducible without downloading space-weather history.
    atmosphere = ADCS.NRLMSISE00(f107=150.0, f107a=150.0, ap=4.0)
    environment = ADCS.Environment(atmosphere=atmosphere)
    initial = ADCS.Orbital_State.from_TLE(
        TLE[0], TLE[1], environment=environment
    )
    start_time = initial.J2000
    end_time = start_time + PROPAGATION_HOURS * 3600.0 * TimeConstants.sec2cent
    steps = int(round(PROPAGATION_HOURS * 3600.0 / DT_SECONDS))

    drag_satellite = make_satellite(include_drag=True)
    no_drag_satellite = make_satellite(include_drag=False)

    sgp4_drag = ADCS.SGP4Orbit(
        TLE,
        end_time=end_time,
        dt=DT_SECONDS,
        satellite=drag_satellite,
        environment=environment,
    )
    sgp4_no_drag = ADCS.SGP4Orbit(
        TLE,
        end_time=end_time,
        dt=DT_SECONDS,
        bstar=0.0,
        environment=environment,
    )
    sgp4_calibrated = ADCS.SGP4Orbit(
        TLE,
        end_time=end_time,
        dt=DT_SECONDS,
        bstar=CALIBRATED_BSTAR,
        environment=environment,
    )

    rk4_drag_a = propagate_rk4(initial, drag_satellite, steps=steps)
    rk4_no_drag_a = propagate_rk4(initial, no_drag_satellite, steps=steps)

    sgp4_drag_a = [semimajor_axis(sgp4_drag.get_os(float(t))) for t in sgp4_drag.times]
    sgp4_no_drag_a = [
        semimajor_axis(sgp4_no_drag.get_os(float(t))) for t in sgp4_no_drag.times
    ]
    sgp4_calibrated_a = [
        semimajor_axis(sgp4_calibrated.get_os(float(t)))
        for t in sgp4_calibrated.times
    ]
    drag_model = drag_satellite.disturbances[0]
    rk4_attitude_projected_cd_area = []
    for time in sgp4_no_drag.times:
        velocity_body = sgp4_no_drag.get_os(float(time)).V
        velocity_body /= np.linalg.norm(velocity_body)
        rk4_attitude_projected_cd_area.append(
            np.sum(
                drag_model.CDs
                * drag_model.areas
                * np.maximum(drag_model.normals @ velocity_body, 0.0)
            )
        )
    # Positive values mean drag lowered the orbit relative to that propagator's
    # own no-drag baseline, which removes most periodic altitude variation.
    sgp4_decay = np.asarray(sgp4_no_drag_a) - np.asarray(sgp4_drag_a)
    sgp4_calibrated_decay = np.asarray(sgp4_no_drag_a) - np.asarray(sgp4_calibrated_a)
    rk4_decay = np.asarray(rk4_no_drag_a) - np.asarray(rk4_drag_a)
    elapsed_hours = (sgp4_drag.times - start_time) * TimeConstants.cent2sec / 3600.0

    print(f"Initial altitude: {np.linalg.norm(initial.R) - EarthConstants.R_e:.2f} km")
    initial_speed_m_s = np.linalg.norm(initial.V) * 1000.0
    initial_drag_force = drag_satellite.dist_forces(
        ADCS.State.from_array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        initial,
    )
    initial_cd_area = (
        2.0 * np.linalg.norm(initial_drag_force)
        / (initial.rho * initial_speed_m_s**2)
    )
    print(f"Initial NRLMSISE density: {initial.rho:.3e} kg/m³")
    print(f"RK4 initial projected CD·A: {initial_cd_area:.3f} m²")
    print(
        "Mean projected CD·A for fixed RK4 attitude: "
        f"{np.mean(rk4_attitude_projected_cd_area):.3f} m²"
    )
    print(f"Spacecraft mass in both cases: {MASS_KG:.1f} kg")
    print(f"SGP4 geometry-averaged B*: {sgp4_drag.bstar:.6e} 1/Earth radii")
    print(f"Diagnostic calibrated B*: {CALIBRATED_BSTAR:.6e} 1/Earth radii")
    print(f"SGP4 drag-induced reduction: {sgp4_decay[-1]:.3f} km")
    print(
        "SGP4 calibrated-B* reduction: "
        f"{sgp4_calibrated_decay[-1]:.3f} km"
    )
    print(f"RK4Orbit drag-induced reduction: {rk4_decay[-1]:.3f} km")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        elapsed_hours,
        sgp4_decay,
        label="SGP4Orbit, geometry-averaged B*",
    )
    ax.plot(
        elapsed_hours,
        sgp4_calibrated_decay,
        linestyle="--",
        label=f"SGP4Orbit, diagnostic B*={CALIBRATED_BSTAR:g}",
    )
    ax.plot(
        elapsed_hours,
        rk4_decay,
        label="RK4Orbit, NRLMSISE + fixed-attitude drag",
    )
    ax.set_title("Drag-induced orbital decay: SGP4Orbit vs RK4Orbit")
    ax.set_xlabel("Elapsed time [hours]")
    ax.set_ylabel("Semi-major-axis reduction from no-drag case [km]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
