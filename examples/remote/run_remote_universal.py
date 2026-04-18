from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ADCS as ADCS


def build_controller():
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    return ADCS.controller.MTQ_w_RW_LP(
        est_sat=real_sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.0, 0.0, 0.0]),
    )


def build_attitude_estimator():
    dt = 20.0

    gyro_noise = ADCS.Noise(std_noise=3.1623e-7)
    sens = [ADCS.Gyro(axis, noise=gyro_noise) for axis in np.eye(3)]

    mtm_noise = ADCS.Noise(std_noise=5e-8)
    sens += [ADCS.MTM(axis, noise=mtm_noise) for axis in np.eye(3)]

    sun_noise = ADCS.Noise(std_noise=2e-3)
    sens += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise) for axis in np.eye(3)]

    gg_dist = [ADCS.disturbances.GG_Disturbance()]
    satellite = ADCS.Satellite(mass=3000, J_0=np.diag([500, 1500, 1500]), sensors=sens, disturbances=gg_dist)

    est_gyro_noise = ADCS.Noise(std_noise=5e-7)
    est_sens = [ADCS.Gyro(axis, noise=est_gyro_noise) for axis in np.eye(3)]

    est_mtm_noise = ADCS.Noise(std_noise=5e-8)
    est_sens += [ADCS.MTM(axis, noise=est_mtm_noise) for axis in np.eye(3)]

    est_sun_noise = ADCS.Noise(std_noise=1e-3)
    est_sens += [ADCS.SunPair(axis, efficiency=0.3, noise=est_sun_noise) for axis in np.eye(3)]

    est_gg_dist = [ADCS.disturbances.GG_Disturbance()]
    est_satellite = ADCS.EstimatedSatellite(
        mass=3200,
        J_0=np.diag([450, 1400, 1400]),
        sensors=est_sens,
        disturbances=est_gg_dist,
    )

    x_hat = np.array([0, 0, 0] + [1, 0, 0, 0])
    P_hat = block_diag(
        np.eye(3) * (0.01) ** 2,
        np.eye(3),
    )
    Q_hat = block_diag(
        np.eye(3) * (1e-8) ** 2.0,
        1e-8 * np.eye(3),
    )

    _ = satellite
    return ADCS.SRUAKF(
        J2000=0.22,
        est_sat=est_satellite,
        x_hat=x_hat,
        P_hat=P_hat,
        Q_hat=Q_hat,
        dt=dt,
        cross_term=True,
        quat_as_vec=False,
    )


def build_orbit_estimator():
    gps_noise = ADCS.Noise(std_noise=np.array([5, 5, 5, 0.1, 0.1, 0.1]))
    gps = [ADCS.GPS(noise=gps_noise)]

    est_gps_noise = ADCS.Noise(std_noise=np.array([3, 3, 3, 0.1, 0.1, 0.1]))
    est_gps = [ADCS.GPS(noise=est_gps_noise)]

    satellite = ADCS.Satellite(mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=gps)
    est_satellite = ADCS.EstimatedSatellite(mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=est_gps)

    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 7.5, 1]),
    )

    P0 = np.diag([500**2.0, 500**2.0, 500**2.0, 0.5**2.0, 0.5**2.0, 0.5**2.0])
    Q0 = np.diag([1, 1, 1, 10, 10, 10])

    _ = satellite
    return ADCS.Orbit_EKF(est_sat=est_satellite, J2000=0.22, os_hat=os0, P_hat=P0, Q_hat=Q0, dt=20.0)


BUILDERS = {
    "controller": build_controller,
    "attitude_estimator": build_attitude_estimator,
    "orbit_estimator": build_orbit_estimator,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a universal ADCS remote component server.")
    parser.add_argument(
        "--component",
        choices=sorted(BUILDERS),
        action="append",
        default=None,
        help="Component type to serve. May be repeated; defaults to controller, attitude_estimator, and orbit_estimator.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("ADCS_REMOTE_BIND_HOST", "0.0.0.0"),
        help="Bind host for the XML-RPC server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ADCS_REMOTE_PORT", "5000")),
        help="Bind port for the XML-RPC server.",
    )
    args = parser.parse_args()

    selected_components = args.component or ["controller", "attitude_estimator", "orbit_estimator"]
    selected_components = list(dict.fromkeys(selected_components))

    controller = BUILDERS["controller"]() if "controller" in selected_components else None
    attitude_estimator = BUILDERS["attitude_estimator"]() if "attitude_estimator" in selected_components else None
    orbit_estimator = BUILDERS["orbit_estimator"]() if "orbit_estimator" in selected_components else None

    print(
        "[remote-runner] starting on "
        f"{args.host}:{args.port} "
        f"with controller={'yes' if controller is not None else 'no'}, "
        f"attitude_estimator={'yes' if attitude_estimator is not None else 'no'}, "
        f"orbit_estimator={'yes' if orbit_estimator is not None else 'no'}"
    )
    ADCS.remote.serve_remote_components(
        controller=controller,
        estimator=attitude_estimator,
        orbit_estimator=orbit_estimator,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
