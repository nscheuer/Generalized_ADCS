"""Simple one-state detumble example using the BDot controller.

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/ADCS_subsystem/controllers/example_bdot.py``
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    BDot,
    Magnetorquer,
    Satellite,
    SatelliteGeometry,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import NoGoal
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Magnetometer
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State
from GenADCS.backends.shared.plotting import (
    BodyAngularRatePlot,
    CommandedMagneticDipolePlot,
    plot,
)


def main(show: bool = True):
    dt = 0.25
    tf = 1800.0

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.15, 0.12, 0.08)),
        panels=[],
    )
    magnetorquers = [
        Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5),
    ]
    magnetometer = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-16 * np.eye(3),
        bias=np.zeros(3),
    )
    adcs = ADCSSubsystem(
        magnetorquers=magnetorquers,
        sensors=[magnetometer],
        N=4,
        state_machine=ADCSStateMachine(
            initial="detumble",
            modes=[ADCSMode("detumble", goal=NoGoal(), controller=BDot(gain=5e4))],
        ),
    )
    satellite = Satellite(geometry=geometry, adcs=adcs)
    environment = Environment(planetary_body=Earth(), magnetic_field=IGRF14())
    dynamics = DynamicsConfig(forces=[], torques=[])

    state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0),
            rates=(0.08, -0.06, 0.05),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 5_382.9, 5_382.9),
            julian_date=2_461_000.5,
        ),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=dt)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=dt)),
        tf=tf,
        progress=True,
    )

    result = simulate(
        satellite,
        environment,
        dynamics,
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(
            N=4, satellite_rate=True, commanded_magnetic_dipole=True
        ),
    )
    rates = result.satellites["sat_000"].satellite_rate
    print(
        f"BDot detumble: ||w|| {np.linalg.norm(rates[0]):.4f} -> "
        f"{np.linalg.norm(rates[-1]):.4f} rad/s"
    )
    plot(
        result,
        BodyAngularRatePlot(),
        CommandedMagneticDipolePlot(),
        layout=(2, 1),
        title="BDot Detumble",
        show=show,
    )
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result


if __name__ == "__main__":
    main()
