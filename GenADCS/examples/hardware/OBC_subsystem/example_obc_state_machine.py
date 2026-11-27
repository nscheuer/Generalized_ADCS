"""OBC + ADCS state machines, FDIR, Goal telecommands, and orbital swath InFOV example.

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/OBC_subsystem/example_obc_state_machine.py``
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.dynamics import StaticGravityGradientModel
from GenADCS.backends.python.environment import DE421, Earth, Environment
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    AngularRateMonitor,
    Completion,
    FDIR,
    InFOV,
    Mode,
    OBCStateMachine,
    OBCSubsystem,
    Satellite,
    SatelliteGeometry,
    Telecommand,
    Transition,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import (
    CoordinateGoal,
    NadirGoal,
    NoGoal,
    SunGoal,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import (
    GoalTable,
    prepare_adcs_sm,
)
from GenADCS.backends.python.hardware.OBC_subsystem.runtime import prepare_obc
from GenADCS.backends.python.helpers.math.frames.conversions import ecef_to_gcrf_matrix
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def build_adcs() -> ADCSSubsystem:
    """Configure the ADCS subsystem with its own mode state machine."""
    adcs_sm = ADCSStateMachine(
        initial="detumble",
        modes=[
            ADCSMode("detumble", goal=NoGoal(), controller="bdot"),
            ADCSMode("nominal", goal=SunGoal(), controller="pd"),
            ADCSMode("safe", goal=SunGoal(), controller="safe_rate"),
        ],
    )
    return ADCSSubsystem(N=2, state_machine=adcs_sm)


def build_obc(target_lat: float = 0.0, target_lon: float = -45.0) -> OBCSubsystem:
    """Configure OBC state machine with Goal telecommands, InFOV swath, and FDIR."""
    swath = InFOV(lat=target_lat, lon=target_lon, max_off_nadir_deg=15.0)
    target_goal = CoordinateGoal(lat=target_lat, lon=target_lon)

    state_machine = OBCStateMachine(
        initial="checkout",
        modes=[
            Mode("checkout", bindings={"adcs": "detumble"}),
            Mode(
                "nominal",
                initial_submode="idle",
                bindings={"adcs": "nominal"},
                submodes=[
                    Mode("idle", bindings={"adcs": ("nominal", SunGoal())}),
                    Mode("payload", bindings={"adcs": ("nominal", target_goal)}),
                    Mode("processing", duration=30.0, auto_target="idle"),
                    Mode("downlink", duration=40.0, auto_target="idle", bindings={"adcs": ("nominal", NadirGoal())}),
                ],
            ),
            Mode("maintenance"),
            Mode("safe_mode", bindings={"adcs": "safe"}),
        ],
        transitions=[
            Transition("checkout", "nominal", Telecommand("ENTER_NOMINAL")),
            # Telecommand carrying a specific Goal override while staying in nominal.idle.
            Transition("nominal.idle", "nominal.idle", Telecommand("POINT_AT")),
            # Automatic orbital/nadir swath entry & exit while in nominal.idle.
            Transition("nominal.idle", "nominal.payload", swath),
            Transition("nominal.payload", "nominal.idle", ~swath),
            # Timed submodes reverting automatically to idle.
            Transition("nominal.idle", "nominal.processing", Telecommand("START_PROCESSING")),
            Transition("nominal.processing", "nominal.idle", Completion()),
            Transition("nominal.idle", "nominal.downlink", Telecommand("START_DOWNLINK")),
            Transition("nominal.downlink", "nominal.idle", Completion()),
            # Maintenance & safe-mode recovery.
            Transition("nominal.*", "maintenance", Telecommand("ENTER_MAINTENANCE")),
            Transition("safe_mode", "checkout", Telecommand("RECOVER_CHECKOUT")),
        ],
        timeline=[
            (20.0, "ENTER_NOMINAL"),
            (40.0, "POINT_AT", NadirGoal()),
            (60.0, "START_PROCESSING"),
            # Around t=105-120s, satellite is in nominal.idle and passes over target_lon.
            (125.0, "START_DOWNLINK"),
            (300.0, "ENTER_MAINTENANCE"),
            (420.0, "RECOVER_CHECKOUT"),
        ],
    )

    fdir = FDIR(
        monitors=[AngularRateMonitor(max_rate=0.032)],
        safe_mode="safe_mode",
    )
    return OBCSubsystem(state_machine=state_machine, fdir=fdir, N=4)


def main() -> None:
    dt = 0.25
    tf = 450.0
    jd0 = 2_461_000.5

    # Place ground target along the equatorial orbit ~108 s ahead of epoch.
    theta_pass = 108.0 * (7_612.6 / 6_878_137.0)
    r_pass_eci = np.array([6_878_137.0 * np.cos(theta_pass), 6_878_137.0 * np.sin(theta_pass), 0.0])
    jd_pass = jd0 + 108.0 / 86400.0
    r_pass_ecef = ecef_to_gcrf_matrix(jd_pass, jd_pass).T @ r_pass_eci
    target_lon = float(np.degrees(np.arctan2(r_pass_ecef[1], r_pass_ecef[0])))

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.18, 0.09, 0.04)),
        panels=[],
    )
    adcs = build_adcs()
    obc = build_obc(target_lat=0.0, target_lon=target_lon)
    satellite = Satellite(geometry=geometry, adcs=adcs, obc=obc)

    environment = Environment(planetary_body=Earth(), ephemeris=DE421())
    dynamics = DynamicsConfig(forces=[], torques=[StaticGravityGradientModel(geometry)])

    state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0),
            rates=(0.002, 0.025, 0.002),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=jd0,
        ),
    )

    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=dt)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=dt)),
        tf=tf,
    )

    start = perf_counter()
    result = simulate(
        satellite,
        environment,
        dynamics,
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(
            N=4, satellite_rate=True, obc_mode=True, adcs_mode=True, adcs_goal=True
        ),
    )
    elapsed = perf_counter() - start

    sat_res = result.satellites["sat_000"]
    goal_table = GoalTable()
    adcs_rt = prepare_adcs_sm(adcs.state_machine, goal_table)
    obc_rt = prepare_obc(obc, adcs_rt, goal_table)
    _, _, goal_names = goal_table.to_arrays()

    times = sat_res.times_s
    obc_ids = sat_res.obc_mode[:, 0].astype(int)
    adcs_ids = sat_res.adcs_mode[:, 0].astype(int)
    goal_ids = sat_res.adcs_goal[:, 0].astype(int)

    print(f"Target lon: {target_lon:.2f} deg | Elapsed: {elapsed:.3f} s | samples: {len(times)}")
    print("State transitions (OBC | ADCS | Goal):")
    prev = (-1, -1, -1)
    for t, oid, aid, gid in zip(times, obc_ids, adcs_ids, goal_ids):
        cur = (oid, aid, gid)
        if cur != prev:
            print(
                f"  t = {t:6.2f} s | OBC: {obc_rt.mode_names[oid]:20s} "
                f"| ADCS: {adcs_rt.mode_names[aid]:10s} | Goal: {goal_names[gid]}"
            )
            prev = cur


if __name__ == "__main__":
    main()
