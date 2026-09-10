"""Shared 6U baseline configuration for estimator-in-the-loop MC campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import ADCS
from ADCS.helpers.math_helpers import normalize
from ADCS.satellite_hardware.errors import AnisotropicNoise
from ADCS.orbits.universal_constants import EarthConstants


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DT_S = 1.0
ORBIT_ALTITUDE_KM = 400.0
ORBIT_INCLINATION_RAD = np.deg2rad(97.0)
ORBIT_PERIOD_S = 5554.0
WHEEL_TORQUE_MAX = 2.0e-3
WHEEL_MOMENTUM_MAX = 15.0e-3
WHEEL_BASELINE_MOMENTUM = 0.05 * WHEEL_MOMENTUM_MAX
MTQ_DIPOLE_MAX = 0.6
KP = 2.9e-4
KD = 8.7e-3
MOMENTUM_GAIN = 1.0e-3


def _sensors(*, estimated: bool) -> list:
    """Build independent true/estimated sensor stacks at the 1 Hz control rate."""
    gyro_arw = np.deg2rad(0.15) / np.sqrt(3600.0)
    gyro_bias_rw = np.deg2rad(0.30) / 3600.0
    gyro_noise = ADCS.Noise(std_noise=gyro_arw)
    gyro_bias = ADCS.Bias(bias=0.0, std_bias=gyro_bias_rw)
    sensors = [
        ADCS.Gyro(axis, sample_time=DT_S, bias=gyro_bias, noise=gyro_noise,
                  estimate_bias=estimated)
        for axis in np.eye(3)
    ]
    # Three MTMs provide the body-frame magnetic field required by MTQ allocation.
    sensors += [
        ADCS.MTM(axis, sample_time=DT_S, noise=ADCS.Noise(std_noise=50e-9))
        for axis in np.eye(3)
    ]
    # The StarTracker model supports Sun exclusion. Earth exclusion is not exposed
    # by the present sensor API, so its 25 deg requirement is recorded but cannot
    # be enforced without extending the flight sensor model.
    for boresight in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
        tracker = ADCS.StarTracker(
            boresight=boresight,
            sample_time=DT_S,
            anisotropic_noise=AnisotropicNoise(
                std_cross=np.deg2rad(5.0 / 3600.0),
                std_roll=np.deg2rad(30.0 / 3600.0),
            ),
            fov=np.deg2rad(20.0),
            sun_exclusion=np.deg2rad(30.0),
        )
        # The LP/QP controller's generic sensor-matrix builder assumes every
        # sensor provides an axis matrix. A star tracker emits a 3-vector, so
        # provide a 3x3 placeholder solely to preserve stacked-measurement indices.
        tracker.axis = np.eye(3)
        sensors.append(tracker)
    return sensors


def make_satellite(number_rw: int, *, estimated: bool) -> ADCS.Satellite:
    """Create the no-disturbance 6U spacecraft for the requested architecture."""
    if number_rw not in (0, 1, 3):
        raise ValueError("number_rw must be 0, 1, or 3.")
    actuators = [ADCS.MTQ(axis, max_torque=MTQ_DIPOLE_MAX) for axis in np.eye(3)]
    wheel_axes = np.eye(3)[2:3] if number_rw == 1 else np.eye(3)[:number_rw]
    actuators += [
        ADCS.RW(
            axis=axis,
            max_torque=WHEEL_TORQUE_MAX,
            J=1.0e-5,
            h=WHEEL_BASELINE_MOMENTUM,
            h_max=WHEEL_MOMENTUM_MAX,
        )
        for axis in wheel_axes
    ]
    satellite_type = ADCS.EstimatedSatellite if estimated else ADCS.Satellite
    return satellite_type(
        mass=12.0,
        J_0=np.diag([0.13, 0.10, 0.05]),
        actuators=actuators,
        sensors=_sensors(estimated=estimated),
        boresight=np.array([0.0, 0.0, 1.0]),
        # Deliberately omit all disturbances for this baseline campaign.
    )


def random_400km_97deg_orbit(rng: np.random.Generator) -> ADCS.Orbital_State:
    """Sample a circular 400 km, 97 deg-inclination orbit uniformly in RAAN and phase."""
    raan = rng.uniform(0.0, 2.0 * np.pi)
    argument_of_latitude = rng.uniform(0.0, 2.0 * np.pi)
    radius = EarthConstants.R_e + ORBIT_ALTITUDE_KM
    circular_speed = np.sqrt(EarthConstants.mu_e / radius)

    r_pqw = radius * np.array([np.cos(argument_of_latitude), np.sin(argument_of_latitude), 0.0])
    v_pqw = circular_speed * np.array([-np.sin(argument_of_latitude), np.cos(argument_of_latitude), 0.0])
    c_raan, s_raan = np.cos(raan), np.sin(raan)
    c_inc, s_inc = np.cos(ORBIT_INCLINATION_RAD), np.sin(ORBIT_INCLINATION_RAD)
    rotation = np.array([
        [c_raan, -s_raan * c_inc, s_raan * s_inc],
        [s_raan, c_raan * c_inc, -c_raan * s_inc],
        [0.0, s_inc, c_inc],
    ])
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22, R=rotation @ r_pqw, V=rotation @ v_pqw
    )


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    return normalize(rng.standard_normal(4))


def make_mc_config(number_rw: int) -> ADCS.MCConfig:
    baseline_h = np.full(number_rw, WHEEL_BASELINE_MOMENTUM)
    return ADCS.MCConfig(
        q=_random_quaternion,
        h=lambda _rng: baseline_h.copy(),
        goal=lambda rng: ADCS.goals.Fixed_Attitude_Goal(_random_quaternion(rng)),
        orbit=random_400km_97deg_orbit,
    )


def make_estimator(
    est_satellite: ADCS.EstimatedSatellite,
    number_rw: int,
    q_hat: np.ndarray | None = None,
) -> ADCS.AugmentedSRUKF:
    """Create an augmented SRUKF with gyro-bias states enabled."""
    estimator_state = ADCS.EstimatorState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]) if q_hat is None else q_hat,
        h=np.full(number_rw, WHEEL_BASELINE_MOMENTUM),
        sens_bias=np.zeros(est_satellite.att_sens_bias_len),
        cov=np.diag(
            np.r_[
                np.full(3, np.deg2rad(1.0) ** 2),
                np.full(3, np.deg2rad(10.0) ** 2),
                np.full(number_rw, (0.1 * WHEEL_MOMENTUM_MAX) ** 2),
                np.full(est_satellite.att_sens_bias_len, np.deg2rad(0.30 / 3600.0) ** 2),
            ]
        ),
        int_cov=np.diag(
            np.r_[
                np.full(3, 1e-12),
                np.full(3, 1e-10),
                np.full(number_rw, 1e-12),
                np.full(est_satellite.att_sens_bias_len, np.deg2rad(0.30 / 3600.0) ** 2),
            ]
        ),
    )
    return ADCS.AugmentedSRUKF(
        est_satellite,
        estimator_state,
        dt=DT_S,
        quaternion_mode="rotation_vector",
    )


def controller_for(allocator: str, est_satellite: ADCS.EstimatedSatellite, number_rw: int):
    h_target = np.zeros(3)
    if number_rw == 1:
        h_target[2] = WHEEL_BASELINE_MOMENTUM
    elif number_rw == 3:
        h_target[:] = WHEEL_BASELINE_MOMENTUM
    controller_type = ADCS.controller.MTQ_w_RW_QP if allocator == "qp" else ADCS.controller.MTQ_w_RW_LP
    return controller_type(
        est_sat=est_satellite,
        p_gain=KP,
        d_gain=KD,
        c_gain=MOMENTUM_GAIN,
        h_target=h_target,
    )


def _save_diagnostic(path: Path, results: ADCS.SimulationResults, title: str, *plots, layout: tuple[int, int]) -> None:
    """Render a package diagnostic figure to disk and leave it open."""
    ADCS.plot(results, *plots, layout=layout, figsize=(10, 7), title=title)
    figure = plt.gcf()
    figure.savefig(path, dpi=220, bbox_inches="tight")


def save_diagnostics(results: ADCS.SimulationResults, label: str) -> None:
    """Save standard package diagnostics for a single closed-loop run."""
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_attitude.png", results, f"{label}: attitude convergence",
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]), layout=(1, 1),
    )
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_state_estimation.png", results, f"{label}: estimator convergence",
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        layout=(2, 1),
    )
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_control_target.png", results, f"{label}: control and target",
        ADCS.plots.ControlPlot(),
        ADCS.plots.TargetPlot(modes=["real_est"]),
        layout=(2, 1),
    )
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_convergence_angle.png", results,
        f"{label}: convergence angle to goal",
        ADCS.plots.TargetPlot(modes=["real_target"]), layout=(1, 1),
    )
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_sensors_bias.png", results, f"{label}: measurements and gyro-bias estimate",
        ADCS.plots.SensorsPlot(sources=["real", "clean"]),
        ADCS.plots.BiasPlot(sources=["real", "estimated"]),
        layout=(2, 1),
    )
    _save_diagnostic(
        OUTPUT_DIR / f"{label}_magnetic_orbit.png", results, f"{label}: magnetic environment",
        ADCS.plots.OrbitMagneticPlotCombined(), layout=(1, 1),
    )


def run_diagnostic(*, number_rw: int, allocator: str, label: str) -> None:
    """Run one seeded closed-loop simulation and save diagnostics; no MC is launched."""
    parser = argparse.ArgumentParser(description=f"Run one 6U {label} diagnostic simulation.")
    parser.add_argument("--tf", type=float, default=ORBIT_PERIOD_S, help="Simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=20260911, help="Random scenario seed.")
    args = parser.parse_args()
    if args.tf <= 0.0:
        raise ValueError("--tf must be positive.")

    rng = np.random.default_rng(args.seed)
    satellite = make_satellite(number_rw, estimated=False)
    est_satellite = make_satellite(number_rw, estimated=True)
    q_initial = _random_quaternion(rng)
    state = ADCS.State(
        w=np.zeros(3), q=q_initial,
        h=np.full(number_rw, WHEEL_BASELINE_MOMENTUM),
    )
    results = ADCS.simulate(
        x=state,
        satellite=satellite,
        est_satellite=est_satellite,
        controller=controller_for(allocator, est_satellite, number_rw),
        estimator=make_estimator(est_satellite, number_rw, q_hat=q_initial),
        goal=ADCS.goals.Fixed_Attitude_Goal(_random_quaternion(rng)),
        os0=random_400km_97deg_orbit(rng),
        dt=DT_S,
        tf=args.tf,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"6u_baseline_3mtq_{number_rw}rw_{allocator}_single"
    result_path = results.save(stem, out_dir=OUTPUT_DIR)
    save_diagnostics(results, stem)
    print(f"Saved one-run result to {result_path}")
    print("Diagnostic figures are open; close them or press Ctrl+C to exit.")
    plt.show(block=True)


def load_diagnostic(*, number_rw: int, allocator: str, label: str) -> None:
    """Load one saved .sim result, save its diagnostics, and keep figures open."""
    parser = argparse.ArgumentParser(description=f"Load 6U {label} diagnostic results.")
    parser.add_argument("--file", type=Path, default=None, help="Specific .sim file to load.")
    args = parser.parse_args()

    if args.file is None:
        prefix = f"6u_baseline_3mtq_{number_rw}rw_{allocator}_single_"
        candidates = sorted(OUTPUT_DIR.glob(f"{prefix}*.sim"))
        if not candidates:
            raise FileNotFoundError(f"No saved .sim files matching {prefix!r} in {OUTPUT_DIR}")
        result_path = candidates[-1]
    else:
        requested_path = args.file.expanduser()
        if not requested_path.is_absolute() and not requested_path.exists():
            requested_path = OUTPUT_DIR / requested_path
        result_path = requested_path.resolve()
        if not result_path.exists():
            raise FileNotFoundError(result_path)

    results = ADCS.SimulationResults.load(result_path, ephem=ADCS.Ephemeris())
    stem = f"6u_baseline_3mtq_{number_rw}rw_{allocator}_loaded"
    save_diagnostics(results, stem)
    print(f"Loaded {result_path}")
    print("Diagnostic figures are open; close them or press Ctrl+C to exit.")
    plt.show(block=True)


def run_campaign(*, number_rw: int, allocator: str, label: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run the 6U {label} Monte Carlo campaign.")
    parser.add_argument("--runs", type=int, default=100, help="Monte Carlo trials (default: 100).")
    parser.add_argument("--tf", type=float, default=ORBIT_PERIOD_S, help="Simulation duration in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument("--seed", type=int, default=20260911, help="Base random seed.")
    args = parser.parse_args()
    if args.runs <= 0 or args.tf <= 0.0 or args.workers <= 0:
        raise ValueError("--runs, --tf, and --workers must be positive.")

    satellite = make_satellite(number_rw, estimated=False)
    est_satellite = make_satellite(number_rw, estimated=True)
    state = ADCS.State(
        w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.full(number_rw, WHEEL_BASELINE_MOMENTUM),
    )
    results = ADCS.simulate_mc(
        x=state,
        satellite=satellite,
        est_satellite=est_satellite,
        controller=controller_for(allocator, est_satellite, number_rw),
        estimator=make_estimator(est_satellite, number_rw),
        goal=ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0])),
        os0=random_400km_97deg_orbit(np.random.default_rng(args.seed)),
        dt=DT_S,
        tf=args.tf,
        mc_config=make_mc_config(number_rw),
        num_runs=args.runs,
        max_workers=args.workers,
        base_seed=args.seed,
    )
    path = results.save(f"6u_baseline_3mtq_{number_rw}rw_{allocator}_mc_{args.runs}", out_dir=OUTPUT_DIR)
    print(f"Saved {len(results)} runs to {path}")
