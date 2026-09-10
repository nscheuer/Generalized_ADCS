"""Run the Section IV XML-RPC worker on a Raspberry Pi.

Start this script on the Raspberry Pi, for example::

    python IV_rpc_worker.py --host 0.0.0.0 --port 5000

The host script is :mod:`IV_rpc_host`.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))

import ADCS


def build_controller():
    satellite = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    return ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5e-5,
        d_gain=2e-3,
        c_gain=1e-3,
        h_target=np.zeros(3),
    )


def build_attitude_estimator():
    dt = 20.0
    gyro_noise = ADCS.Noise(std_noise=3.1623e-7)
    sensors = [ADCS.Gyro(axis, noise=gyro_noise) for axis in np.eye(3)]
    sensors += [ADCS.MTM(axis, noise=ADCS.Noise(std_noise=5e-8)) for axis in np.eye(3)]
    sensors += [
        ADCS.SunPair(axis, efficiency=0.3, noise=ADCS.Noise(std_noise=2e-3))
        for axis in np.eye(3)
    ]
    satellite = ADCS.Satellite(
        mass=3000,
        J_0=np.diag([500, 1500, 1500]),
        sensors=sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )

    est_sensors = [ADCS.Gyro(axis, noise=ADCS.Noise(std_noise=5e-7)) for axis in np.eye(3)]
    est_sensors += [ADCS.MTM(axis, noise=ADCS.Noise(std_noise=5e-8)) for axis in np.eye(3)]
    est_sensors += [
        ADCS.SunPair(axis, efficiency=0.3, noise=ADCS.Noise(std_noise=1e-3))
        for axis in np.eye(3)
    ]
    estimated_satellite = ADCS.EstimatedSatellite(
        mass=3200,
        J_0=np.diag([450, 1400, 1400]),
        sensors=est_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_hat = ADCS.EstimatorState(w=np.zeros(3), q=[1, 0, 0, 0])
    P_hat = block_diag(np.eye(3) * 0.01**2, np.eye(3))
    Q_hat = block_diag(np.eye(3) * 1e-8**2, 1e-8 * np.eye(3))
    _ = satellite
    return ADCS.AugmentedSRUKF(
        J2000=0.22,
        est_sat=estimated_satellite,
        x_hat=x_hat,
        P_hat=P_hat,
        Q_hat=Q_hat,
        dt=dt,
        cross_term=True,
        quat_as_vec=False,
    )


def build_orbit_estimator():
    gps = [ADCS.GPS(noise=ADCS.Noise(std_noise=np.array([5, 5, 5, 0.1, 0.1, 0.1])))]
    est_gps = [ADCS.GPS(noise=ADCS.Noise(std_noise=np.array([3, 3, 3, 0.1, 0.1, 0.1])))]
    satellite = ADCS.Satellite(mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=gps)
    estimated_satellite = ADCS.EstimatedSatellite(
        mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=est_gps
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 1]),
    )
    P0 = np.diag([500**2, 500**2, 500**2, 0.5**2, 0.5**2, 0.5**2])
    Q0 = np.diag([1, 1, 1, 10, 10, 10])
    _ = satellite
    return ADCS.Orbit_EKF(
        est_sat=estimated_satellite, J2000=0.22, os_hat=os0,
        P_hat=P0, Q_hat=Q0, dt=20.0,
    )


BUILDERS = {
    "controller": build_controller,
    "attitude_estimator": build_attitude_estimator,
    "orbit_estimator": build_orbit_estimator,
}


def _detect_primary_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Section IV ADCS XML-RPC worker.")
    parser.add_argument("--component", choices=sorted(BUILDERS), action="append", default=None)
    parser.add_argument("--host", default=os.getenv("ADCS_REMOTE_BIND_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ADCS_REMOTE_PORT", "5000")))
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.component or list(BUILDERS)))
    controller = BUILDERS["controller"]() if "controller" in selected else None
    estimator = BUILDERS["attitude_estimator"]() if "attitude_estimator" in selected else None
    orbit_estimator = BUILDERS["orbit_estimator"]() if "orbit_estimator" in selected else None
    connect_host = _detect_primary_ip() if args.host in {"0.0.0.0", "::"} else args.host

    print(f"[section-IV-worker] serving on {args.host}:{args.port}")
    print(f"[section-IV-worker] set ADCS_REMOTE_HOST={connect_host}")
    print(f"[section-IV-worker] set ADCS_REMOTE_PORT={args.port}")
    try:
        ADCS.remote.serve_remote_components(
            controller=controller,
            estimator=estimator,
            orbit_estimator=orbit_estimator,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\n[section-IV-worker] stopped")


if __name__ == "__main__":
    main()
