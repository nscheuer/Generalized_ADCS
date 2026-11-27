"""Subsystem-level ADCS state machine and compiled Goal table."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import NamedTuple, cast

import numpy as np
from numba import njit

from .controllers import Controller, NoController
from .controllers.runtime import ControllerTable
from .goals._orbital_kernels import _reference as _orbital_goal_ref
from .goals.coordinate_goal import _coordinate_fmc_rate, _coordinate_reference
from .goals import (
    AlongTrackGoal,
    AngularRateGoal,
    CoordinateFMCGoal,
    CoordinateGoal,
    CrossTrackGoal,
    Goal,
    NadirGoal,
    NoGoal,
    QuaternionGoal,
    SunGoal,
    VelocityGoal,
    ZenithGoal,
)

# Compiled Goal kind opcodes.
GOAL_NONE = 0
GOAL_NADIR = 1
GOAL_ZENITH = 2
GOAL_ALONG_TRACK = 3
GOAL_CROSS_TRACK = 4
GOAL_VELOCITY = 5
GOAL_SUN = 6
GOAL_COORDINATE = 7
GOAL_COORDINATE_FMC = 8
GOAL_QUATERNION = 9
GOAL_ANGULAR_RATE = 10

_GOAL_PARAM_WIDTH = 4


class GoalTable:
    """Registry that flattens heterogeneous ``Goal`` objects into numeric arrays."""

    def __init__(self) -> None:
        self.kinds: list[int] = [GOAL_NONE]
        self.params: list[np.ndarray] = [np.zeros(_GOAL_PARAM_WIDTH, dtype=np.float64)]
        self.names: list[str] = ["NoGoal"]

    def register(self, goal: Goal | None) -> int:
        if goal is None or isinstance(goal, NoGoal):
            return 0
        p = np.zeros(_GOAL_PARAM_WIDTH, dtype=np.float64)
        if isinstance(goal, NadirGoal):
            kind = GOAL_NADIR
            label = "NadirGoal"
        elif isinstance(goal, ZenithGoal):
            kind = GOAL_ZENITH
            label = "ZenithGoal"
        elif isinstance(goal, AlongTrackGoal):
            kind = GOAL_ALONG_TRACK
            label = "AlongTrackGoal"
        elif isinstance(goal, CrossTrackGoal):
            kind = GOAL_CROSS_TRACK
            label = "CrossTrackGoal"
        elif isinstance(goal, VelocityGoal):
            kind = GOAL_VELOCITY
            label = "VelocityGoal"
        elif isinstance(goal, SunGoal):
            kind = GOAL_SUN
            label = "SunGoal"
        elif isinstance(goal, CoordinateFMCGoal):
            kind = GOAL_COORDINATE_FMC
            p[0:3] = goal.target_ecef_m
            p[3] = float(goal.fmc_level)
            label = f"CoordinateFMCGoal({goal.lat_deg:.2f}, {goal.lon_deg:.2f}, fmc={goal.fmc_level})"
        elif isinstance(goal, CoordinateGoal):
            kind = GOAL_COORDINATE
            p[0:3] = goal.target_ecef_m
            label = f"CoordinateGoal({goal.lat_deg:.2f}, {goal.lon_deg:.2f})"
        elif isinstance(goal, QuaternionGoal):
            kind = GOAL_QUATERNION
            p[0:4] = goal.quaternion
            label = "QuaternionGoal"
        elif isinstance(goal, AngularRateGoal):
            kind = GOAL_ANGULAR_RATE
            p[0:3] = goal.rate
            label = "AngularRateGoal"
        else:
            raise TypeError(f"unsupported Goal in state machine: {type(goal).__name__}")
        idx = len(self.kinds)
        self.kinds.append(kind)
        self.params.append(p)
        self.names.append(label)
        return idx

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        return (
            np.asarray(self.kinds, dtype=np.int64),
            np.vstack(self.params),
            tuple(self.names),
        )


@dataclass(slots=True)
class ADCSMode:
    """One ADCS operating mode (e.g. ``detumble``, ``nominal``, ``safe``)."""

    name: str
    goal: Goal = field(default_factory=NoGoal)
    controller: Controller | str = field(default_factory=NoController)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ADCSMode name must be nonempty")


@dataclass(slots=True)
class ADCSStateMachine:
    """Subsystem-level ADCS state machine (standalone or OBC-commanded)."""

    initial: str
    modes: list[ADCSMode]
    timeline: list[tuple[float, str | Goal]] = field(
        default_factory=lambda: cast(list[tuple[float, str | Goal]], [])
    )

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("ADCSStateMachine requires at least one ADCSMode")
        names = [m.name for m in self.modes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate ADCSMode names")
        if self.initial not in names:
            raise ValueError(f"initial ADCS mode {self.initial!r} not found")
        for t, item in self.timeline:
            if not math.isfinite(t) or t < 0.0:
                raise ValueError("ADCS timeline times must be finite and nonnegative")
            if isinstance(item, str) and item not in names:
                raise ValueError(f"unknown ADCS mode in timeline: {item!r}")


class ADCSSMRuntime(NamedTuple):
    """Compiled Numba tables for the ADCS subsystem state machine."""

    enabled: bool
    initial_mode: int
    mode_default_goal: np.ndarray
    mode_default_controller: np.ndarray
    # Standalone timeline (terminated by inf / -1):
    tl_times: np.ndarray
    tl_modes: np.ndarray
    tl_goals: np.ndarray
    # Unified Goal table:
    goal_kinds: np.ndarray
    goal_params: np.ndarray
    # Python-side metadata:
    mode_names: tuple[str, ...]
    goal_names: tuple[str, ...]


def prepare_adcs_sm(
    adcs_sm: ADCSStateMachine | None,
    goal_table: GoalTable | None = None,
    controller_table: ControllerTable | None = None,
) -> ADCSSMRuntime:
    """Flatten ``ADCSStateMachine`` into Numba arrays."""
    if goal_table is None:
        goal_table = GoalTable()
    if controller_table is None:
        controller_table = ControllerTable()
    if adcs_sm is None:
        g_kinds, g_params, g_names = goal_table.to_arrays()
        return ADCSSMRuntime(
            enabled=False,
            initial_mode=0,
            mode_default_goal=np.zeros(1, dtype=np.int64),
            mode_default_controller=np.zeros(1, dtype=np.int64),
            tl_times=np.array([np.inf], dtype=np.float64),
            tl_modes=np.array([-1], dtype=np.int64),
            tl_goals=np.array([-1], dtype=np.int64),
            goal_kinds=g_kinds,
            goal_params=g_params,
            mode_names=("default",),
            goal_names=g_names,
        )

    mode_to_id = {m.name: idx for idx, m in enumerate(adcs_sm.modes)}
    mode_names = tuple(m.name for m in adcs_sm.modes)
    mode_default_goal = np.empty(len(adcs_sm.modes), dtype=np.int64)
    mode_default_controller = np.zeros(len(adcs_sm.modes), dtype=np.int64)
    for idx, m in enumerate(adcs_sm.modes):
        mode_default_goal[idx] = goal_table.register(m.goal)
        if isinstance(m.controller, Controller):
            mode_default_controller[idx] = controller_table.register(m.controller)
        elif m.controller is not None:
            raise TypeError(
                f"ADCSMode {m.name!r} controller must be a Controller instance, "
                f"got {type(m.controller).__name__}"
            )

    sorted_tl = sorted(adcs_sm.timeline, key=lambda item: item[0])
    tl_times = np.empty(len(sorted_tl) + 1, dtype=np.float64)
    tl_modes = np.empty(len(sorted_tl) + 1, dtype=np.int64)
    tl_goals = np.empty(len(sorted_tl) + 1, dtype=np.int64)
    for i, (t, item) in enumerate(sorted_tl):
        tl_times[i] = float(t)
        if isinstance(item, str):
            tl_modes[i] = mode_to_id[item]
            tl_goals[i] = mode_default_goal[tl_modes[i]]
        else:
            tl_modes[i] = -1
            tl_goals[i] = goal_table.register(item)
    tl_times[-1] = np.inf
    tl_modes[-1] = -1
    tl_goals[-1] = -1

    g_kinds, g_params, g_names = goal_table.to_arrays()
    return ADCSSMRuntime(
        enabled=True,
        initial_mode=mode_to_id[adcs_sm.initial],
        mode_default_goal=mode_default_goal,
        mode_default_controller=mode_default_controller,
        tl_times=tl_times,
        tl_modes=tl_modes,
        tl_goals=tl_goals,
        goal_kinds=g_kinds,
        goal_params=g_params,
        mode_names=mode_names,
        goal_names=g_names,
    )


def init_adcs_work(runtime: ADCSSMRuntime) -> np.ndarray:
    """Allocate mutable ADCS SM state ``[active_mode, active_goal, tl_idx]``."""
    work = np.empty(3, dtype=np.int64)
    work[0] = runtime.initial_mode
    work[1] = runtime.mode_default_goal[runtime.initial_mode]
    work[2] = 0
    return work


@njit(cache=True)
def step_adcs_sm(
    elapsed_s: float,
    adcs_sm: ADCSSMRuntime,
    work: np.ndarray,
    cmd_mode: int,
    cmd_goal: int,
) -> tuple[int, int]:
    """Update ADCS mode and active goal from OBC command or standalone timeline."""
    mode = work[0]
    goal = work[1]

    # 1. Apply OBC-commanded mode/goal if provided.
    if cmd_mode >= 0 and cmd_mode != mode:
        mode = cmd_mode
        goal = adcs_sm.mode_default_goal[mode]
    if cmd_goal >= 0:
        goal = cmd_goal

    # 2. Apply standalone ADCS timeline events if due.
    tl_idx = work[2]
    while elapsed_s >= adcs_sm.tl_times[tl_idx]:
        next_m = adcs_sm.tl_modes[tl_idx]
        next_g = adcs_sm.tl_goals[tl_idx]
        if next_m >= 0:
            mode = next_m
        if next_g >= 0:
            goal = next_g
        tl_idx += 1
    work[2] = tl_idx
    work[0] = mode
    work[1] = goal
    return mode, goal


@njit(cache=True)
def evaluate_goal(
    goal_idx: int,
    goal_kinds: np.ndarray,
    goal_params: np.ndarray,
    state: np.ndarray,
    epoch_jd: float,
    runtime,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate active goal index into ``(q_goal(4), w_goal(3))`` (NaN when unset)."""
    nan_q = np.full(4, np.nan, dtype=np.float64)
    nan_w = np.full(3, np.nan, dtype=np.float64)
    if goal_idx < 0 or goal_idx >= goal_kinds.size:
        return nan_q, nan_w
    kind = goal_kinds[goal_idx]
    if kind == GOAL_NONE:
        return nan_q, nan_w
    if kind >= GOAL_NADIR and kind <= GOAL_SUN:
        return _orbital_goal_ref(state, epoch_jd, runtime, kind - 1)
    if kind == GOAL_COORDINATE:
        return _coordinate_reference(
            state, epoch_jd, goal_params[goal_idx, :3], runtime.mu
        )
    if kind == GOAL_COORDINATE_FMC:
        return _coordinate_fmc_rate(
            state,
            epoch_jd,
            goal_params[goal_idx, :3],
            runtime.mu,
            int(goal_params[goal_idx, 3]),
        )
    if kind == GOAL_QUATERNION:
        return goal_params[goal_idx, :4].copy(), nan_w
    if kind == GOAL_ANGULAR_RATE:
        return nan_q, goal_params[goal_idx, :3].copy()
    return nan_q, nan_w
