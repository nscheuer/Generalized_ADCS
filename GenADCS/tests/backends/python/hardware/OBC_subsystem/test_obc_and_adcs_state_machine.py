"""Fast unit and integration tests for OBC and ADCS state machines."""

from __future__ import annotations

import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Earth, Environment, IGRF14
from GenADCS.backends.python.hardware import (
    ADCSMode,
    ADCSStateMachine,
    ADCSSubsystem,
    After,
    AngularRateMonitor,
    AttitudeController,
    BDot,
    Completion,
    FDIR,
    FDIRTriggered,
    InFOV,
    L2Allocator,
    MRPFeedbackLinearized,
    Magnetorquer,
    Mode,
    NoController,
    OBCStateMachine,
    OBCSubsystem,
    Passthrough,
    ReactionWheel,
    Satellite,
    SatelliteGeometry,
    Telecommand,
    Transition,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope, Magnetometer
from GenADCS.backends.python.hardware.ADCS_subsystem.goals import (
    CoordinateGoal,
    NadirGoal,
    NoGoal,
    QuaternionGoal,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import (
    GoalTable,
    init_adcs_work,
    prepare_adcs_sm,
    step_adcs_sm,
)
from GenADCS.backends.python.hardware.OBC_subsystem.runtime import (
    init_obc_work,
    prepare_obc,
    step_obc,
)
from GenADCS.backends.python.helpers.math.frames.conversions import ecef_to_gcrf_matrix
from GenADCS.backends.python.propagation import (
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.simulate import ResultsFormat, simulate
from GenADCS.backends.python.solver import AttitudeSolver, OrbitalSolver, Solver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State, WheelMomentumState


def _make_state_vec(
    rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pos: tuple[float, float, float] = (6_878_137.0, 0.0, 0.0),
    vel: tuple[float, float, float] = (0.0, 7_612.6, 0.0),
) -> np.ndarray:
    y = np.zeros(13, dtype=np.float64)
    y[0] = 1.0
    y[4:7] = rates
    y[7:10] = pos
    y[10:13] = vel
    return y


def test_adcs_sm_rejects_string_controller() -> None:
    sm = ADCSStateMachine(
        initial="detumble",
        modes=[ADCSMode("detumble", goal=NoGoal(), controller="bdot")],
    )
    with pytest.raises(TypeError, match="Controller instance"):
        prepare_adcs_sm(sm)


def test_obc_telecommand_same_step_preserves_adcs_binding_and_goal() -> None:
    goal_table = GoalTable()
    adcs_sm = ADCSStateMachine(
        initial="detumble",
        modes=[
            ADCSMode("detumble", goal=NoGoal(), controller=NoController()),
            ADCSMode("nominal", goal=NadirGoal(), controller=NoController()),
        ],
    )
    adcs_rt = prepare_adcs_sm(adcs_sm, goal_table)
    q_override = QuaternionGoal((0.0, 1.0, 0.0, 0.0))
    obc = OBCSubsystem(
        state_machine=OBCStateMachine(
            initial="boot",
            modes=[
                Mode("boot", bindings={"adcs": "detumble"}),
                Mode("nominal", bindings={"adcs": ("nominal", NadirGoal())}),
                Mode("logging"),  # No ADCS binding (-1)
            ],
            transitions=[
                Transition("boot", "nominal", Telecommand("GO_NOMINAL", goal=q_override)),
                Transition("nominal", "logging", Telecommand("START_LOG")),
            ],
            timeline=[(1.0, "GO_NOMINAL"), (1.0, "START_LOG")],
        )
    )
    obc_rt = prepare_obc(obc, adcs_rt, goal_table)
    work = init_obc_work(obc_rt)
    y = _make_state_vec()

    state, cm, cg = step_obc(y, 1.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[state] == "logging"
    assert adcs_rt.mode_names[cm] == "nominal"
    assert goal_table.names[cg] == "QuaternionGoal"


def test_obc_after_and_completion_transitions() -> None:
    goal_table = GoalTable()
    adcs_rt = prepare_adcs_sm(
        ADCSStateMachine(
            initial="idle",
            modes=[ADCSMode("idle", goal=NoGoal(), controller=NoController())],
        ),
        goal_table,
    )
    obc = OBCSubsystem(
        state_machine=OBCStateMachine(
            initial="warmup",
            modes=[
                Mode("warmup"),
                Mode("work", duration=3.0),
                Mode("done"),
            ],
            transitions=[
                Transition("warmup", "work", After(2.0)),
                Transition("work", "done", Completion()),
            ],
        )
    )
    obc_rt = prepare_obc(obc, adcs_rt, goal_table)
    work = init_obc_work(obc_rt)
    y = _make_state_vec()

    s0, _, _ = step_obc(y, 0.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s0] == "warmup"
    s1, _, _ = step_obc(y, 2.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s1] == "work"
    s2, _, _ = step_obc(y, 4.9, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s2] == "work"
    s3, _, _ = step_obc(y, 5.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s3] == "done"


def test_obc_infov_entry_and_exit() -> None:
    jd = 2_461_000.5
    r_eci = np.array([6_878_137.0, 0.0, 0.0])
    r_ecef = ecef_to_gcrf_matrix(jd, jd).T @ r_eci
    subsat_lon = float(np.degrees(np.arctan2(r_ecef[1], r_ecef[0])))
    swath = InFOV(lat=0.0, lon=subsat_lon, max_off_nadir_deg=15.0)

    goal_table = GoalTable()
    adcs_rt = prepare_adcs_sm(
        ADCSStateMachine(
            initial="idle",
            modes=[ADCSMode("idle", goal=NoGoal(), controller=NoController())],
        ),
        goal_table,
    )
    obc = OBCSubsystem(
        state_machine=OBCStateMachine(
            initial="idle",
            modes=[Mode("idle"), Mode("payload")],
            transitions=[
                Transition("idle", "payload", swath),
                Transition("payload", "idle", ~swath),
            ],
        )
    )
    obc_rt = prepare_obc(obc, adcs_rt, goal_table)
    work = init_obc_work(obc_rt)

    # Directly over target -> enters payload with CoordinateGoal.
    y_over = _make_state_vec(pos=(6_878_137.0, 0.0, 0.0))
    s_in, _, cg = step_obc(y_over, 0.0, jd, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s_in] == "payload"
    assert isinstance(CoordinateGoal(0.0, subsat_lon), CoordinateGoal)
    assert "CoordinateGoal" in goal_table.names[cg]

    # Far side of orbit -> exits payload back to idle.
    y_far = _make_state_vec(pos=(-6_878_137.0, 0.0, 0.0))
    s_out, _, _ = step_obc(y_far, 1.0, jd, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s_out] == "idle"


def test_obc_fdir_triggered_custom_target_and_unlatch() -> None:
    goal_table = GoalTable()
    adcs_rt = prepare_adcs_sm(
        ADCSStateMachine(
            initial="nominal",
            modes=[
                ADCSMode("nominal", goal=NadirGoal(), controller=NoController()),
                ADCSMode("safe", goal=NoGoal(), controller=NoController()),
            ],
        ),
        goal_table,
    )
    obc = OBCSubsystem(
        state_machine=OBCStateMachine(
            initial="nominal",
            modes=[
                Mode("nominal", bindings={"adcs": "nominal"}),
                Mode("alt_safe", bindings={"adcs": "safe"}),
                Mode("safe_mode", bindings={"adcs": "safe"}),
            ],
            transitions=[
                Transition("nominal", "alt_safe", FDIRTriggered(monitor="rate_mon")),
                Transition("alt_safe", "nominal", After(2.0)),
            ],
        ),
        fdir=FDIR(monitors=[AngularRateMonitor(max_rate=0.05, name="rate_mon")]),
    )
    obc_rt = prepare_obc(obc, adcs_rt, goal_table)
    work = init_obc_work(obc_rt)

    # Fault trips FDIRTriggered transition -> alt_safe, latched.
    y_fast = _make_state_vec(rates=(0.1, 0.0, 0.0))
    s1, cm1, _ = step_obc(y_fast, 0.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s1] == "alt_safe"
    assert adcs_rt.mode_names[cm1] == "safe"
    assert work[3] == 1.0

    # Auto-recovery after 2.0 s unlatches FDIR even though alt_safe != fdir_safe_state.
    y_calm = _make_state_vec(rates=(0.0, 0.0, 0.0))
    s2, cm2, _ = step_obc(y_calm, 2.0, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s2] == "nominal"
    assert adcs_rt.mode_names[cm2] == "nominal"
    assert work[3] == 0.0

    # Second fault re-arms and triggers again.
    s3, _, _ = step_obc(y_fast, 2.5, 2_461_000.5, 0.5, obc_rt, work)
    assert obc_rt.mode_names[s3] == "alt_safe"
    assert work[3] == 1.0


def test_obc_fdir_triggered_unknown_monitor_raises() -> None:
    obc = OBCSubsystem(
        state_machine=OBCStateMachine(
            initial="nominal",
            modes=[Mode("nominal"), Mode("safe_mode")],
            transitions=[Transition("nominal", "safe_mode", FDIRTriggered(monitor="missing"))],
        ),
        fdir=FDIR(monitors=[AngularRateMonitor(max_rate=0.05, name="rate_mon")]),
    )
    with pytest.raises(ValueError, match="unknown monitor"):
        prepare_obc(obc)


def test_simulate_applies_initial_obc_binding_at_t0() -> None:
    adcs = ADCSSubsystem(
        N=1,
        state_machine=ADCSStateMachine(
            initial="detumble",
            modes=[
                ADCSMode("detumble", goal=NoGoal(), controller=NoController()),
                ADCSMode("point", goal=NadirGoal(), controller=NoController()),
            ],
        ),
    )
    obc = OBCSubsystem(
        N=1,
        state_machine=OBCStateMachine(
            initial="science",
            modes=[
                Mode("science", bindings={"adcs": ("point", QuaternionGoal((1.0, 0.0, 0.0, 0.0)))}),
                Mode("safe_mode", bindings={"adcs": "detumble"}),
            ],
            transitions=[Transition("science", "safe_mode", Telecommand("SAFE"))],
            timeline=[(1.0, "SAFE")],
        ),
    )
    sat = Satellite(
        geometry=SatelliteGeometry(mass=5.0, inertia=np.eye(3), panels=[]),
        adcs=adcs,
        obc=obc,
    )
    state = State(
        attitude=AttitudeState(orientation=(1.0, 0.0, 0.0, 0.0), rates=(0.0, 0.0, 0.0)),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
        ),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=0.5)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=0.5)),
        tf=1.5,
    )
    res = simulate(
        sat,
        Environment(planetary_body=Earth()),
        DynamicsConfig(forces=[], torques=[]),
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(obc_mode=True, adcs_mode=True, adcs_goal=True),
    )
    sat_res = res.satellites["sat_000"]
    # t=0: OBC=science(0), ADCS=point(1), Goal=QuaternionGoal(2)
    assert int(sat_res.obc_mode[0, 0]) == 0
    assert int(sat_res.adcs_mode[0, 0]) == 1
    assert int(sat_res.adcs_goal[0, 0]) == 2
    # t=1.0: OBC=safe_mode(1), ADCS=detumble(0), Goal=NoGoal(0)
    assert int(sat_res.obc_mode[2, 0]) == 1
    assert int(sat_res.adcs_mode[2, 0]) == 0
    assert int(sat_res.adcs_goal[2, 0]) == 0


def test_state_machine_toggles_sensors_and_actuators_and_telecommands() -> None:
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
    gyro = Gyroscope(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=1e-12 * np.eye(3),
        bias=np.zeros(3),
    )
    mag = Magnetometer(
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=1.0,
        noise=1e-16 * np.eye(3),
        bias=np.zeros(3),
    )
    att_ctrl = AttitudeController(
        feedback=MRPFeedbackLinearized(kp=0.05, kd=0.1),
        feedforward=Passthrough(),
        allocator=L2Allocator(),
    )
    adcs = ADCSSubsystem(
        reaction_wheels=wheels,
        magnetorquers=mtqs,
        sensors=[gyro, mag],
        N=1,
        state_machine=ADCSStateMachine(
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
                    "nominal",
                    goal=NadirGoal(),
                    controller=att_ctrl,
                    sensors={"*": True},
                    actuators={"ReactionWheel": True, "Magnetorquer": True},
                ),
                ADCSMode(
                    "safe",
                    goal=NoGoal(),
                    controller=BDot(gain=5e4),
                    sensors={"Gyroscope": False, "Magnetometer": True},
                    actuators={"ReactionWheel": False, "Magnetorquer": True},
                ),
            ],
        ),
    )
    obc = OBCSubsystem(
        N=1,
        state_machine=OBCStateMachine(
            initial="checkout",
            modes=[
                Mode(
                    "checkout",
                    initial_submode="power_on",
                    submodes=[
                        Mode(
                            "power_on",
                            duration=1.0,
                            bindings={"adcs": "off", "sensors": {"*": False}, "actuators": {"*": False}},
                        ),
                        Mode(
                            "sensors_on",
                            duration=1.0,
                            bindings={"adcs": "off", "sensors": {"Gyroscope": True}},
                        ),
                    ],
                ),
                Mode("nominal", bindings={"adcs": "nominal"}),
                Mode("safe_mode", bindings={"adcs": "safe"}),
            ],
            transitions=[
                Transition("checkout.power_on", "checkout.sensors_on", Completion()),
                Transition("checkout.sensors_on", "nominal", Completion()),
                Transition(
                    "nominal",
                    "nominal",
                    Telecommand("DISABLE_GYRO_AND_RWX", sensors={"Gyroscope": False}, actuators={"rw_x": False}),
                ),
                Transition("nominal", "safe_mode", Telecommand("ENTER_SAFE")),
            ],
            timeline=[
                (3.0, "DISABLE_GYRO_AND_RWX"),
                (4.0, "ENTER_SAFE"),
            ],
        ),
    )
    sat = Satellite(
        geometry=SatelliteGeometry(mass=5.0, inertia=np.eye(3), panels=[]),
        adcs=adcs,
        obc=obc,
    )
    state = State(
        attitude=AttitudeState(orientation=(0.9, 0.1, 0.2, 0.1), rates=(0.05, -0.04, 0.03)),
        orbit=OrbitalState.from_cartesian(
            position=(6_878_137.0, 0.0, 0.0),
            velocity=(0.0, 7_612.6, 0.0),
            julian_date=2_461_000.5,
        ),
        internal=WheelMomentumState(wheel_momentum=np.zeros(3)),
    )
    solver = Solver(
        orbital=OrbitalSolver(KeplerianOrbitalPropagator(dt=0.5)),
        attitude=AttitudeSolver(RK4AttitudePropagator(dt=0.5)),
        tf=5.0,
    )
    res = simulate(
        sat,
        Environment(planetary_body=Earth(), magnetic_field=IGRF14()),
        DynamicsConfig(forces=[], torques=[]),
        state,
        solver,
        backend="Python",
        resultsformat=ResultsFormat(
            obc_mode=True,
            adcs_mode=True,
            gyroscope_reading=True,
            magnetometer_reading=True,
            commanded_wheel_torque=True,
            commanded_magnetic_dipole=True,
        ),
    )
    sat_res = res.satellites["sat_000"]
    # dt=0.5 -> index k corresponds to t = 0.5*k.
    # t=0.5 (idx 1): checkout.power_on -> all sensors off (NaN), all actuators 0.0
    assert np.all(np.isnan(sat_res.gyroscope_reading[1]))
    assert np.all(np.isnan(sat_res.magnetometer_reading[1]))
    assert np.all(sat_res.commanded_wheel_torque[1] == 0.0)
    assert np.all(sat_res.commanded_magnetic_dipole[1] == 0.0)

    # t=1.5 (idx 3): checkout.sensors_on -> gyro on (finite), mag still off (NaN)
    assert np.all(np.isfinite(sat_res.gyroscope_reading[3]))
    assert np.all(np.isnan(sat_res.magnetometer_reading[3]))

    # t=2.5 (idx 5): nominal -> all sensors finite, all 3 RWs active
    assert np.all(np.isfinite(sat_res.gyroscope_reading[5]))
    assert np.all(np.isfinite(sat_res.magnetometer_reading[5]))
    assert np.any(sat_res.commanded_wheel_torque[5] != 0.0)

    # t=3.5 (idx 7): after DISABLE_GYRO_AND_RWX -> gyro NaN, mag finite, rw_x == 0.0
    assert np.all(np.isnan(sat_res.gyroscope_reading[7]))
    assert np.all(np.isfinite(sat_res.magnetometer_reading[7]))
    assert sat_res.commanded_wheel_torque[7, 0] == 0.0
    assert np.any(sat_res.commanded_wheel_torque[7, 1:3] != 0.0)

    # t=4.5 (idx 9): safe_mode -> gyro NaN, mag finite, all RWs 0.0, MTQs active
    assert np.all(np.isnan(sat_res.gyroscope_reading[9]))
    assert np.all(np.isfinite(sat_res.magnetometer_reading[9]))
    assert np.all(sat_res.commanded_wheel_torque[9] == 0.0)
    assert np.any(sat_res.commanded_magnetic_dipole[9] != 0.0)
