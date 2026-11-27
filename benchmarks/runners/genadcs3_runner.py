"""Generalized ADCS 3.0 runner (run inside the GenADCS 3.0 virtual environment).

``--variant`` selects the backend: ``python`` (Numba-compiled), ``cpp``
(pybind11) or ``rust`` (PyO3). All three use ``CoupledRK4Propagator``, which
integrates attitude and orbit together, so the "orbit" scenario runs the same
coupled model. Attitude, rate, position and velocity are recorded every step.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np

from common import S, emit, parse_args, timed_runs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
warnings.filterwarnings("ignore")

BACKEND_NAME = {"python": "Python", "cpp": "C", "rust": "Rust"}
LABEL = {"python": "Python (Numba)", "cpp": "C++", "rust": "Rust"}


def make_run_factory(variant: str):
    def module(name):
        return import_module(f"GenADCS.backends.{variant}.{name}")

    states, hardware, env = module("state"), module("hardware"), module("environment")
    config, propagation, solvers = module("config"), module("propagation"), module("solver")
    dynamics, simulate = module("dynamics"), module("simulate").simulate
    ResultsFormat = import_module(
        "GenADCS.backends.python.helpers.input_output.resultsformat").ResultsFormat

    def make_run(steps: int):
        geometry = hardware.SatelliteGeometry(
            mass=S.MASS, inertia=np.diag(S.INERTIA_DIAG), panels=[])
        satellite = hardware.Satellite(geometry=geometry)
        environment = env.Environment(planetary_body=env.Earth(), atmosphere=env.NRLMSISE00(),
                                      ephemeris=env.DE421())
        dyn = config.DynamicsConfig(
            forces=[dynamics.ZonalForceModel(geometry, environment, degrees=(2,))], torques=[])
        state = states.State(
            attitude=states.AttitudeState(orientation=S.Q0, rates=S.W0),
            orbit=states.OrbitalState.from_cartesian(position=S.R0, velocity=S.V0,
                                                     julian_date=S.JULIAN_DATE0),
        )
        solver = solvers.CoupledSolver(propagation.CoupledRK4Propagator(dt=S.DT),
                                       tf=steps * S.DT, progress=False)
        fmt = ResultsFormat(satellite_position=True, satellite_velocity=True)

        def run():
            result = simulate(satellite, environment, dyn, state, solver,
                              backend=BACKEND_NAME[variant], resultsformat=fmt)
            fields = result.satellites["sat_000"].fields
            q = np.asarray(fields["satellite_attitude"][-1], dtype=float)
            return {
                "r": np.asarray(fields["satellite_position"][-1], dtype=float).tolist(),
                "v": np.asarray(fields["satellite_velocity"][-1], dtype=float).tolist(),
                "dcm_bi": S.quat_b_from_i_to_dcm(q.tolist()),
                "w": np.asarray(fields["satellite_rate"][-1], dtype=float).tolist(),
                "recorded": len(result.satellites["sat_000"].times_s),
            }
        return run
    return make_run


def main():
    args = parse_args(variants=list(BACKEND_NAME))
    samples, final = timed_runs(make_run_factory(args.variant), args)
    emit(tool="Generalized ADCS 3.0", variant=LABEL[args.variant], version=version("GenADCS"),
         scenario=args.scenario, steps=args.steps, samples_s=samples, final=final,
         notes=["CoupledRK4Propagator: attitude and orbit integrated together.",
                "Every step records attitude, rate, position and velocity."])


if __name__ == "__main__":
    main()
