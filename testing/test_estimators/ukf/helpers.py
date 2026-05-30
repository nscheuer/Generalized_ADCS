from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.satellite_hardware.errors import AnisotropicNoise, Bias, Noise
from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Gyro,
    MTM,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite


def seed(value: int = 0) -> None:
    np.random.seed(value)


def make_orbital_state(
    *,
    j2000: float = 0.22,
    R: np.ndarray | None = None,
    V: np.ndarray | None = None,
    B: np.ndarray | None = None,
    S: np.ndarray | None = None,
    rho: float = 5.0e-12,
    sunlit: bool = True,
) -> Orbital_State:
    os = Orbital_State(
        ephem=Ephemeris(),
        J2000=j2000,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else np.asarray(R, dtype=float),
        V=np.array([0.0, 7.5, 0.0]) if V is None else np.asarray(V, dtype=float),
        B=np.array([2.0e-5, 1.0e-5, -3.0e-5]) if B is None else np.asarray(B, dtype=float),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]) if S is None else np.asarray(S, dtype=float),
        rho=rho,
    )
    os.is_sunlit = lambda sunlit=sunlit: sunlit
    return os


def make_orbital_sequence(
    *,
    count: int,
    dt: float,
    base: Orbital_State | None = None,
) -> list[Orbital_State]:
    base_os = make_orbital_state() if base is None else base
    sequence: list[Orbital_State] = []
    for index in range(count):
        os = base_os.copy()
        os.J2000 = base_os.J2000 + index * dt * TimeConstants.sec2cent
        sequence.append(os)
    return sequence


def make_state(
    *,
    w: np.ndarray | None = None,
    q: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> np.ndarray:
    omega = np.zeros(3) if w is None else np.asarray(w, dtype=float)
    quat = np.array([1.0, 0.0, 0.0, 0.0]) if q is None else normalize(np.asarray(q, dtype=float))
    if h is None:
        return np.concatenate([omega, quat])
    return np.concatenate([omega, quat, np.asarray(h, dtype=float)])


def quat_error_deg(q_true: np.ndarray, q_est: np.ndarray) -> float:
    dot = np.clip(abs(float(np.dot(q_true, q_est))), -1.0, 1.0)
    return float(2.0 * np.arccos(dot) * 180.0 / np.pi)


def make_mtqs(*, estimate_bias: bool = False, std_noise: float = 1.0e-5) -> list[MTQ]:
    noise = Noise(noise=0.0, std_noise=std_noise)
    bias = Bias(bias=0.0, std_bias=0.0) if estimate_bias else None
    return [
        MTQ(axis=axis, max_torque=1.0, noise=noise.copy(), bias=bias.copy() if bias else None, estimate_bias=estimate_bias)
        for axis in MathConstants.unitvecs
    ]


def make_rws(*, h: float = 1.0, estimate_bias: bool = False) -> list[RW]:
    torque_noise = Noise(noise=0.0, std_noise=1.0e-5)
    meas_noise = Noise(noise=0.0, std_noise=1.0e-5)
    bias = Bias(bias=0.0, std_bias=0.0) if estimate_bias else None
    return [
        RW(
            axis=axis,
            max_torque=0.2,
            J=0.02,
            h=h,
            h_max=4.0,
            noise=torque_noise.copy(),
            h_meas_noise=meas_noise.copy(),
            bias=bias.copy() if bias else None,
            estimate_bias=estimate_bias,
        )
        for axis in MathConstants.unitvecs
    ]


def sensor_family(name: str):
    if name == "gyro":
        return [Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-5), estimate_bias=False) for axis in MathConstants.unitvecs]
    if name == "mtm":
        return [MTM(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-8), estimate_bias=False) for axis in MathConstants.unitvecs]
    if name == "sunpair":
        return [SunPair(axis=axis, efficiency=1.0, noise=Noise(noise=0.0, std_noise=1.0e-5), estimate_bias=False) for axis in MathConstants.unitvecs]
    if name == "sunsensor":
        return [SunSensor(axis=axis, efficiency=1.0, noise=Noise(noise=0.0, std_noise=1.0e-5), estimate_bias=False) for axis in MathConstants.unitvecs]
    if name == "earth_horizon":
        return [
            EarthHorizonSensor(
                boresight=np.array([-1.0, 0.0, 0.0]),
                fov=np.deg2rad(140.0),
                noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1.0e-5)),
                estimate_bias=False,
            )
        ]
    if name == "star_tracker":
        return [
            StarTracker(
                boresight=np.array([0.0, 0.0, 1.0]),
                fov=np.deg2rad(120.0),
                sun_exclusion=np.deg2rad(5.0),
                anisotropic_noise=AnisotropicNoise(std_cross=1.0e-6, std_roll=2.0e-6),
                estimate_bias=False,
            )
        ]
    if name == "star_tracker_quaternion":
        return [
            StarTrackerQuaternion(
                boresight=np.array([0.0, 0.0, 1.0]),
                fov=np.deg2rad(140.0),
                sun_exclusion=np.deg2rad(5.0),
                min_stars=2,
                noise=Noise(noise=np.zeros(4), std_noise=np.full(4, 1.0e-6)),
                estimate_bias=False,
            )
        ]
    raise ValueError(f"Unknown sensor family {name!r}")


def make_baseline_sensors(*, gyro_bias: np.ndarray | None = None, estimate_gyro_bias: bool = False) -> list:
    sensors = []
    mtm_noise = Noise(noise=0.0, std_noise=1.0e-8)
    gyro_noise = Noise(noise=0.0, std_noise=1.0e-5)
    sun_noise = Noise(noise=0.0, std_noise=1.0e-5)
    bias_vec = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float)
    for index, axis in enumerate(MathConstants.unitvecs):
        sensors.append(MTM(axis=axis, noise=mtm_noise.copy()))
    for index, axis in enumerate(MathConstants.unitvecs):
        bias = Bias(bias=bias_vec[index], std_bias=0.0) if (estimate_gyro_bias or np.any(bias_vec)) else None
        sensors.append(Gyro(axis=axis, noise=gyro_noise.copy(), bias=bias, estimate_bias=estimate_gyro_bias))
    for axis in MathConstants.unitvecs:
        sensors.append(SunPair(axis=axis, efficiency=1.0, noise=sun_noise.copy()))
    return sensors


def make_satellites(
    *,
    sensors: list,
    actuators: list | None = None,
    estimated_sensors: list | None = None,
    estimated_actuators: list | None = None,
    disturbances: list | None = None,
    estimated_disturbances: list | None = None,
    J_0: np.ndarray | None = None,
) -> tuple[Satellite, EstimatedSatellite]:
    acts = make_mtqs() if actuators is None else actuators
    est_acts = make_mtqs() if estimated_actuators is None else estimated_actuators
    dists = [GG_Disturbance()] if disturbances is None else disturbances
    est_dists = [GG_Disturbance()] if estimated_disturbances is None else estimated_disturbances
    inertia = np.diag([3.4, 2.9, 1.3]) if J_0 is None else np.asarray(J_0, dtype=float)
    real_sat = Satellite(mass=4.0, J_0=inertia, actuators=acts, sensors=sensors, disturbances=dists)
    est_sat = EstimatedSatellite(
        mass=4.0,
        J_0=inertia,
        actuators=est_acts,
        sensors=sensors if estimated_sensors is None else estimated_sensors,
        disturbances=est_dists,
    )
    return real_sat, est_sat


def reduced_process_cov(est_sat: EstimatedSatellite, *, dt: float, rate_std: float = 1.0e-4, att_std: float = 1.0e-4) -> np.ndarray:
    base = block_diag(
        np.eye(3) * rate_std**2,
        np.eye(3) * att_std**2,
        np.eye(est_sat.number_RW) * (1.0e-4) ** 2 if est_sat.number_RW else np.zeros((0, 0)),
        np.eye(est_sat.act_bias_len) * (1.0e-6) ** 2 if est_sat.act_bias_len else np.zeros((0, 0)),
        np.eye(est_sat.att_sens_bias_len) * (1.0e-6) ** 2 if est_sat.att_sens_bias_len else np.zeros((0, 0)),
        np.eye(est_sat.dist_param_len) * (1.0e-6) ** 2 if est_sat.dist_param_len else np.zeros((0, 0)),
    )
    return np.array(base if np.ndim(base) == 2 else np.zeros((0, 0))) * max(dt, 1.0)


def reduced_state_cov(est_sat: EstimatedSatellite) -> np.ndarray:
    parts = [np.eye(3) * (0.02) ** 2, np.eye(3) * (0.15) ** 2]
    if est_sat.number_RW:
        parts.append(np.eye(est_sat.number_RW) * (0.2) ** 2)
    if est_sat.act_bias_len:
        parts.append(np.eye(est_sat.act_bias_len) * (0.05) ** 2)
    if est_sat.att_sens_bias_len:
        parts.append(np.eye(est_sat.att_sens_bias_len) * (0.05) ** 2)
    if est_sat.dist_param_len:
        parts.append(np.eye(est_sat.dist_param_len) * (0.05) ** 2)
    return np.array(block_diag(*parts))


def full_state_cov(est_sat: EstimatedSatellite) -> np.ndarray:
    parts = [np.eye(3) * (0.02) ** 2, np.eye(4) * (0.15) ** 2]
    if est_sat.number_RW:
        parts.append(np.eye(est_sat.number_RW) * (0.2) ** 2)
    if est_sat.act_bias_len:
        parts.append(np.eye(est_sat.act_bias_len) * (0.05) ** 2)
    if est_sat.att_sens_bias_len:
        parts.append(np.eye(est_sat.att_sens_bias_len) * (0.05) ** 2)
    if est_sat.dist_param_len:
        parts.append(np.eye(est_sat.dist_param_len) * (0.05) ** 2)
    return np.array(block_diag(*parts))


def make_estimate_guess(est_sat: EstimatedSatellite, *, with_rw: bool | None = None, with_bias: bool = False) -> np.ndarray:
    include_rw = est_sat.number_RW > 0 if with_rw is None else with_rw
    state = make_state(
        w=np.array([2.0e-3, -1.0e-3, 1.5e-3]),
        q=np.array([0.97, 0.15, -0.08, 0.16]),
        h=np.full(est_sat.number_RW, 0.2) if include_rw and est_sat.number_RW else None,
    )
    pieces = [state]
    if est_sat.act_bias_len:
        pieces.append(np.zeros(est_sat.act_bias_len))
    if est_sat.att_sens_bias_len:
        pieces.append(np.zeros(est_sat.att_sens_bias_len))
    if est_sat.dist_param_len:
        pieces.append(np.zeros(est_sat.dist_param_len))
    return np.concatenate(pieces)


def make_ukf(
    est_sat: EstimatedSatellite,
    *,
    x_hat: np.ndarray | None = None,
    P_hat: np.ndarray | None = None,
    Q_hat: np.ndarray | None = None,
    dt: float = 5.0,
    cross_term: bool = False,
    quat_as_vec: bool = False,
) -> UAKF:
    guess = make_estimate_guess(est_sat) if x_hat is None else np.asarray(x_hat, dtype=float)
    if quat_as_vec:
        P = full_state_cov(est_sat) if P_hat is None else np.asarray(P_hat, dtype=float)
        Q = np.eye(guess.size) * 1.0e-6 if Q_hat is None else np.asarray(Q_hat, dtype=float)
    else:
        P = reduced_state_cov(est_sat) if P_hat is None else np.asarray(P_hat, dtype=float)
        Q = reduced_process_cov(est_sat, dt=dt) if Q_hat is None else np.asarray(Q_hat, dtype=float)
    return UAKF(est_sat=est_sat, J2000=0.22, x_hat=guess, P_hat=P, Q_hat=Q, dt=dt, cross_term=cross_term, quat_as_vec=quat_as_vec)


def measurement_vector(
    sat: Satellite,
    x: np.ndarray,
    os: Orbital_State,
    *,
    noiseless: bool = True,
) -> np.ndarray:
    if noiseless:
        return sat.noiseless_sensor_readings(x=x, os=os)
    return sat.sensor_readings(x=x, os=os)


@dataclass
class SimulationResult:
    truth: np.ndarray
    estimate: np.ndarray
    measurements: np.ndarray
    covariances: list[np.ndarray]


def run_sequence(
    real_sat: Satellite,
    ukf: UAKF,
    *,
    x_true: np.ndarray,
    os_sequence: list[Orbital_State],
    control: np.ndarray | None = None,
    measurement_hook=None,
) -> SimulationResult:
    x = x_true.copy()
    u = np.zeros(len(real_sat.actuators)) if control is None else np.asarray(control, dtype=float)
    truth_hist = []
    est_hist = []
    meas_hist = []
    cov_hist: list[np.ndarray] = []
    for index, os in enumerate(os_sequence[:-1]):
        sensors = measurement_vector(real_sat, x, os, noiseless=False)
        if measurement_hook is not None:
            sensors = measurement_hook(index, sensors.copy(), os, x.copy())
        x_hat = ukf.update(u=u, sensors=sensors, os=os)
        truth_hist.append(x.copy())
        est_hist.append(x_hat.copy())
        meas_hist.append(sensors.copy())
        cov_hist.append(ukf.x_hat.cov.copy())
        x = real_sat.noiseless_rk4(
            x=x.copy(),
            u=u,
            dt=ukf.dt,
            orbital_state0=os,
            orbital_state1=os_sequence[index + 1],
            quat_as_vec=True,
        )
        x[3:7] = normalize(x[3:7])
    return SimulationResult(
        truth=np.asarray(truth_hist),
        estimate=np.asarray(est_hist),
        measurements=np.asarray(meas_hist),
        covariances=cov_hist,
    )
