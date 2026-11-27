"""Multi-day representative mission: checkout, detumble, sun-point, FMC imaging, and ground-station downlink.

Run from the repository root with:
``venv/bin/python GenADCS/examples/hardware/OBC_subsystem/example_obc_state_machine.py``
"""

from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

BACKEND = "Rust"  # "Python", "C", or "Rust"
_BACKEND_PKG = {"Python": "python", "C": "cpp", "Rust": "rust"}[BACKEND]

_config = import_module(f"GenADCS.backends.{_BACKEND_PKG}.config")
_dynamics = import_module(f"GenADCS.backends.{_BACKEND_PKG}.dynamics")
_environment = import_module(f"GenADCS.backends.{_BACKEND_PKG}.environment")
_hardware = import_module(f"GenADCS.backends.{_BACKEND_PKG}.hardware")
_goals = import_module(f"GenADCS.backends.{_BACKEND_PKG}.hardware.ADCS_subsystem.goals")
_sensors = import_module(f"GenADCS.backends.{_BACKEND_PKG}.hardware.ADCS_subsystem.sensors")
_propagation = import_module(f"GenADCS.backends.{_BACKEND_PKG}.propagation")
_simulate = import_module(f"GenADCS.backends.{_BACKEND_PKG}.simulate")
_solver = import_module(f"GenADCS.backends.{_BACKEND_PKG}.solver")
_state = import_module(f"GenADCS.backends.{_BACKEND_PKG}.state")

DynamicsConfig = _config.DynamicsConfig
StaticGravityGradientModel = _dynamics.StaticGravityGradientModel
DE421 = _environment.DE421
Earth = _environment.Earth
Environment = _environment.Environment
IGRF14 = _environment.IGRF14
ADCSMode = _hardware.ADCSMode
ADCSStateMachine = _hardware.ADCSStateMachine
ADCSSubsystem = _hardware.ADCSSubsystem
AngularRateMonitor = _hardware.AngularRateMonitor
AttitudeController = _hardware.AttitudeController
BDot = _hardware.BDot
Completion = _hardware.Completion
FDIR = _hardware.FDIR
InFOV = _hardware.InFOV
L2Allocator = _hardware.L2Allocator
MRPFeedbackLinearized = _hardware.MRPFeedbackLinearized
Magnetorquer = _hardware.Magnetorquer
Mode = _hardware.Mode
NoController = _hardware.NoController
OBCStateMachine = _hardware.OBCStateMachine
OBCSubsystem = _hardware.OBCSubsystem
Passthrough = _hardware.Passthrough
ReactionWheel = _hardware.ReactionWheel
Satellite = _hardware.Satellite
SatelliteGeometry = _hardware.SatelliteGeometry
Telecommand = _hardware.Telecommand
Transition = _hardware.Transition
CoordinateFMCGoal = _goals.CoordinateFMCGoal
CoordinateGoal = _goals.CoordinateGoal
NoGoal = _goals.NoGoal
SunGoal = _goals.SunGoal
AutonomousStarTracker = _sensors.AutonomousStarTracker
Gyroscope = _sensors.Gyroscope
Magnetometer = _sensors.Magnetometer
KeplerianOrbitalPropagator = _propagation.KeplerianOrbitalPropagator
RK4AttitudePropagator = _propagation.RK4AttitudePropagator
ResultsFormat = _simulate.ResultsFormat
simulate = _simulate.simulate
AttitudeSolver = _solver.AttitudeSolver
OrbitalSolver = _solver.OrbitalSolver
Solver = _solver.Solver
AttitudeState = _state.AttitudeState
OrbitalState = _state.OrbitalState
State = _state.State
WheelMomentumState = _state.WheelMomentumState
from GenADCS.backends.shared.plotting import (
    ADCSModePlot,
    AbsolutePointingErrorPlot,
    BodyAngularRatePlot,
    OBCModePlot,
    WheelMomentumPlot,
    plot,
)


def build_adcs(img_goal: CoordinateFMCGoal, gs_goal: CoordinateGoal) -> ADCSSubsystem:
    """Configure the ADCS subsystem with its own mode state machine."""
    wheels = [
        ReactionWheel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1e-3, 0.2, 1.0, name="rw_x"),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e-3, 0.2, 1.0, name="rw_y"),
        ReactionWheel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1e-3, 0.2, 1.0, name="rw_z"),
    ]
    mtqs = [
        Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5),
        Magnetorquer((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5),
    ]
    mag = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-16 * np.eye(3),
        bias=np.zeros(3),
    )
    gyro = Gyroscope(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=1e-12 * np.eye(3),
        bias=np.zeros(3),
    )
    ast = AutonomousStarTracker(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-10 * np.eye(3),
        bias=np.zeros(3),
    )
    att_ctrl = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.01, kd=0.05),
        feedforward=Passthrough(),
        allocator=L2Allocator(),
    )
    adcs_sm = ADCSStateMachine(
        initial="off",
        modes=[
            ADCSMode(
                "off",
                goal=NoGoal(),
                controller=NoController(),
                sensors={"*": False},
                actuators={"*": False},
            ),
            ADCSMode(
                "detumble",
                goal=NoGoal(),
                controller=BDot(gain=5e4),
                sensors={"Magnetometer": True, "Gyroscope": True, "AutonomousStarTracker": False},
                actuators={"Magnetorquer": True, "ReactionWheel": False},
            ),
            ADCSMode(
                "sun_point",
                goal=SunGoal(),
                controller=att_ctrl,
                sensors={"*": True},
                actuators={"ReactionWheel": True, "Magnetorquer": True},
            ),
            ADCSMode("imaging", goal=img_goal, controller=att_ctrl),
            ADCSMode("downlink", goal=gs_goal, controller=att_ctrl),
            ADCSMode(
                "safe",
                goal=NoGoal(),
                controller=BDot(gain=5e4),
                sensors={"Magnetometer": True, "Gyroscope": False, "AutonomousStarTracker": False},
                actuators={"ReactionWheel": False, "Magnetorquer": True},
            ),
        ],
    )
    return ADCSSubsystem(
        reaction_wheels=wheels,
        magnetorquers=mtqs,
        sensors=[mag, gyro, ast],
        N=1,
        state_machine=adcs_sm,
    )


def build_obc(
    img_goal: CoordinateFMCGoal,
    gs_goal: CoordinateGoal,
) -> OBCSubsystem:
    """Configure OBC state machine for a multi-day Earth-observation mission."""
    img_swath = InFOV(lat=img_goal.lat_deg, lon=img_goal.lon_deg, max_off_nadir_deg=25.0, goal=img_goal)
    gs_swath = InFOV(lat=gs_goal.lat_deg, lon=gs_goal.lon_deg, max_off_nadir_deg=25.0, goal=gs_goal)

    state_machine = OBCStateMachine(
        initial="checkout",
        modes=[
            Mode(
                "checkout",
                initial_submode="power_on",
                submodes=[
                    Mode(
                        "power_on",
                        duration=200.0,
                        bindings={"adcs": "off", "sensors": {"*": False}, "actuators": {"*": False}},
                    ),
                    Mode(
                        "sensors_on",
                        duration=200.0,
                        bindings={
                            "adcs": "off",
                            "sensors": {"Magnetometer": True, "Gyroscope": True},
                        },
                    ),
                    Mode(
                        "actuators_on",
                        duration=200.0,
                        bindings={
                            "adcs": "off",
                            "sensors": {"AutonomousStarTracker": True},
                            "actuators": {"Magnetorquer": True},
                        },
                    ),
                ],
            ),
            Mode("detumble", bindings={"adcs": "detumble"}),
            Mode(
                "nominal",
                initial_submode="sun_point",
                bindings={"adcs": "sun_point"},
                submodes=[
                    Mode("sun_point", bindings={"adcs": "sun_point"}),
                    Mode("imaging", bindings={"adcs": ("imaging", img_goal)}),
                    Mode("downlink", bindings={"adcs": ("downlink", gs_goal)}),
                ],
            ),
            Mode(
                "safe_mode",
                bindings={
                    "adcs": "safe",
                    "sensors": {"Gyroscope": False, "AutonomousStarTracker": False},
                    "actuators": {"ReactionWheel": False, "Magnetorquer": True},
                },
            ),
        ],
        transitions=[
            Transition("checkout.power_on", "checkout.sensors_on", Completion()),
            Transition("checkout.sensors_on", "checkout.actuators_on", Completion()),
            Transition("checkout.actuators_on", "detumble", Completion()),
            Transition("detumble", "nominal", Telecommand("ENTER_NOMINAL")),
            Transition("nominal.sun_point", "nominal.imaging", img_swath),
            Transition("nominal.imaging", "nominal.sun_point", ~img_swath),
            Transition("nominal.sun_point", "nominal.downlink", gs_swath),
            Transition("nominal.downlink", "nominal.sun_point", ~gs_swath),
            Transition(
                "nominal.sun_point",
                "nominal.sun_point",
                Telecommand("DISABLE_STAR_TRACKER", sensors={"AutonomousStarTracker": False}),
            ),
            Transition(
                "nominal.sun_point",
                "nominal.sun_point",
                Telecommand("ENABLE_STAR_TRACKER", sensors={"AutonomousStarTracker": True}),
            ),
            Transition(
                "nominal.sun_point",
                "nominal.sun_point",
                Telecommand("DISABLE_RW_X", actuators={"rw_x": False}),
            ),
            Transition(
                "nominal.sun_point",
                "nominal.sun_point",
                Telecommand("ENABLE_RW_X", actuators={"rw_x": True}),
            ),
            Transition("nominal.*", "safe_mode", Telecommand("ENTER_SAFE")),
            Transition("safe_mode", "checkout", Telecommand("RECOVER_CHECKOUT")),
        ],
        timeline=[
            (3_600.0, "ENTER_NOMINAL"),
            (12_000.0, "DISABLE_STAR_TRACKER"),
            (18_000.0, "ENABLE_STAR_TRACKER"),
            (24_000.0, "DISABLE_RW_X"),
            (30_000.0, "ENABLE_RW_X"),
            (80_000.0, "ENTER_SAFE"),
            (86_400.0, "RECOVER_CHECKOUT"),
            (90_000.0, "ENTER_NOMINAL"),
        ],
    )

    fdir = FDIR(
        monitors=[AngularRateMonitor(max_rate=0.5)],
        safe_mode="safe_mode",
    )
    return OBCSubsystem(state_machine=state_machine, fdir=fdir, N=1)


def main(show: bool = True) -> None:
    dt = 2.0
    tf = 2.0 * 86_400.0  # 2-day mission
    jd0 = 2_461_000.5

    img_goal = CoordinateFMCGoal(lat=0.0, lon=0.0, fmc_level=1)
    gs_goal = CoordinateGoal(lat=0.0, lon=180.0)

    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.18, 0.09, 0.04)),
        panels=[],
    )
    adcs = build_adcs(img_goal, gs_goal)
    obc = build_obc(img_goal, gs_goal)
    satellite = Satellite(geometry=geometry, adcs=adcs, obc=obc)

    environment = Environment(planetary_body=Earth(), ephemeris=DE421(), magnetic_field=IGRF14())
    dynamics = DynamicsConfig(forces=[], torques=[StaticGravityGradientModel(geometry)])

    state = State(
        attitude=AttitudeState(
            orientation=(1.0, 0.0, 0.0, 0.0),
            rates=(0.005, -0.004, 0.003),
        ),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=jd0,
        ),
        internal=WheelMomentumState(wheel_momentum=np.zeros(3)),
    )

    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=dt)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=dt)),
        tf=tf,
        progress=True,
    )

    start = perf_counter()
    result = simulate(
        satellite,
        environment,
        dynamics,
        state,
        solver,
        backend=BACKEND,
        resultsformat=ResultsFormat(
            N=15,
            satellite_attitude=True,
            satellite_rate=True,
            satellite_wheel_momenta=True,
            goal_attitude=True,
            obc_mode=True,
            adcs_mode=True,
            adcs_goal=True,
        ),
    )
    elapsed = perf_counter() - start

    sat_res = result.satellites["sat_000"]
    adcs_mode_names = [m.name for m in adcs.state_machine.modes]
    obc_mode_names: list[str] = []

    def _collect_obc_modes(modes) -> None:
        for m in modes:
            obc_mode_names.append(m.name)
            if getattr(m, "submodes", None):
                _collect_obc_modes(m.submodes)

    _collect_obc_modes(obc.state_machine.modes)

    goal_names = ["NoGoal"]

    def _reg_goal(g) -> None:
        if g is not None and type(g).__name__ != "NoGoal":
            goal_names.append(type(g).__name__)

    for m in adcs.state_machine.modes:
        _reg_goal(m.goal)
    if getattr(adcs.state_machine, "timeline", None):
        for _, target in adcs.state_machine.timeline:
            if not isinstance(target, str):
                _reg_goal(target)

    def _collect_obc_goals(modes) -> None:
        for m in modes:
            b = getattr(m, "bindings", None) or {}
            adcs_b = b.get("adcs")
            if isinstance(adcs_b, tuple) and len(adcs_b) == 2:
                _reg_goal(adcs_b[1])
            elif adcs_b is not None and not isinstance(adcs_b, str):
                _reg_goal(adcs_b)
            if getattr(m, "submodes", None):
                _collect_obc_goals(m.submodes)

    _collect_obc_goals(obc.state_machine.modes)
    for tr in obc.state_machine.transitions:
        _reg_goal(getattr(tr.trigger, "goal", None))
    for tl in (getattr(obc.state_machine, "timeline", None) or (), getattr(obc, "timeline", None) or ()):
        for entry in tl:
            if len(entry) >= 3:
                _reg_goal(entry[2])

    times = sat_res.times_s
    obc_ids = sat_res.obc_mode[:, 0].astype(int)
    adcs_ids = sat_res.adcs_mode[:, 0].astype(int)
    goal_ids = sat_res.adcs_goal[:, 0].astype(int)

    print(f"2-day mission ({BACKEND}) | Elapsed: {elapsed:.3f} s | samples: {len(times)}")
    print("State transitions (OBC | ADCS | Goal):")
    prev = (-1, -1, -1)
    for t, oid, aid, gid in zip(times, obc_ids, adcs_ids, goal_ids):
        cur = (oid, aid, gid)
        if cur != prev:
            oname = obc_mode_names[oid] if 0 <= oid < len(obc_mode_names) else str(oid)
            aname = adcs_mode_names[aid] if 0 <= aid < len(adcs_mode_names) else str(aid)
            gname = goal_names[gid] if 0 <= gid < len(goal_names) else str(gid)
            print(
                f"  t = {t:8.1f} s | OBC: {oname:20s} "
                f"| ADCS: {aname:10s} | Goal: {gname}"
            )
            prev = cur

    plot(
        result,
        OBCModePlot(),
        ADCSModePlot(),
        BodyAngularRatePlot(),
        WheelMomentumPlot(),
        AbsolutePointingErrorPlot(),
        layout=(3, 2),
        title="Multi-Day Mission: OBC & ADCS State Machines",
        figure_kwargs={"figsize": (15, 10)},
        show=show,
    )
    if show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
