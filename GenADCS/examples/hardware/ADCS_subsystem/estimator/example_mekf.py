"""Run and plot the Python AttitudeMEKF with Keplerian + RK4 propagation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

# Select the backend for this example. The ADCS estimator integration is
# currently available on Python only.
backend = "python"

from GenADCS.backends.python.hardware.ADCS_subsystem.estimator import AttitudeMEKF
from GenADCS.backends.shared.plotting import (
    AbsoluteKnowledgeErrorQuaternionPlot,
    AbsoluteKnowledgeErrorRatePlot,
    EstimatedAttitudeCovariancePlot,
    EstimatedAttitudePlot,
    EstimatedRateCovariancePlot,
    EstimatedRatePlot,
    KalmanGainPlot,
    MeasurementInnovationCovariancePlot,
    MeasurementInnovationPlot,
    MeasurementResidualPlot,
    NormalizedEstimationErrorSquaredPlot,
    NormalizedInnovationSquaredPlot,
    plot,
)


def run_example(*, backend=backend, progress=False):
    """Run the MEKF example with Keplerian orbit and RK4 attitude propagation."""

    from GenADCS.backends.python.config import DynamicsConfig
    from GenADCS.backends.python.environment import DE421, Earth, Environment
    from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
    from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
        AutonomousStarTracker,
        Gyroscope,
    )
    from GenADCS.backends.python.propagation import (
        KeplerianOrbitalPropagator,
        RK4AttitudePropagator,
    )
    from GenADCS.backends.python.simulate import ResultsFormat, simulate
    from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
    from GenADCS.backends.python.state import AttitudeState, OrbitalState, State

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    true_rate = np.array([0.01, -0.015, 0.02])
    initial_rate = true_rate + np.array([1e-3, -1e-3, 2e-3])
    initial_attitude = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
    gyroscope = Gyroscope(
        np.zeros(3), identity, identity, 10.0, np.eye(3) * 1e-8, np.zeros(3)
    )
    star_tracker = AutonomousStarTracker(
        np.zeros(3), identity, identity, 1.0, np.eye(3) * 1e-6,
        np.zeros(3), fov_half_angle=np.deg2rad(5.0),
    )
    estimator = AttitudeMEKF(
        initial_attitude, initial_rate, np.eye(6) * 1e-2, 1e-8
    )
    satellite = Satellite(
        SatelliteGeometry(12.0, np.diag([0.13, 0.10, 0.05]), []),
        ADCSSubsystem(sensors=[gyroscope, star_tracker], estimator=estimator),
    )
    state = State(
        AttitudeState(identity, true_rate),
        OrbitalState.from_cartesian(
            (6_878_137.0, 0.0, 0.0), (0.0, 7_612.6, 0.0), 2_461_000.5
        ),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=0.25)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=0.25)),
        tf=100_000.0,
        progress=progress,
    )
    results_format = ResultsFormat(
        satellite_attitude=True,
        satellite_rate=True,
        estimated_attitude=True,
        estimated_rate=True,
        estimated_covariance=True,
        estimated_attitude_covariance=True,
        estimated_rate_covariance=True,
        measurement_innovation=True,
        measurement_innovation_covariance=True,
        normalized_innovation_squared=True,
        normalized_estimation_error_squared=True,
        measurement_residual=True,
        kalman_gain=True,
    )
    return simulate(
        satellite,
        Environment(planetary_body=Earth(), ephemeris=DE421()),
        DynamicsConfig(forces=[], torques=[]),
        state,
        solver,
        resultsformat=results_format,
        seed=7,
    )


def main():
    result = run_example(backend=backend, progress=True)
    plot(
        result,
        EstimatedAttitudePlot(),
        EstimatedRatePlot(),
        EstimatedAttitudeCovariancePlot(option="individual"),
        EstimatedRateCovariancePlot(option="individual"),
        AbsoluteKnowledgeErrorQuaternionPlot(option="magnitude"),
        AbsoluteKnowledgeErrorRatePlot(option="magnitude"),
        MeasurementInnovationPlot(option="individual"),
        MeasurementInnovationCovariancePlot(option="trace"),
        MeasurementResidualPlot(option="individual"),
        NormalizedInnovationSquaredPlot(),
        NormalizedEstimationErrorSquaredPlot(),
        KalmanGainPlot(option="trace"),
        layout=(4, 3),
        title="AttitudeMEKF: Keplerian Orbit and RK4 Attitude Propagation",
        show=True,
    )
    import matplotlib.pyplot as plt

    plt.show()
    return result


if __name__ == "__main__":
    main()
