"""Compiled Numba runtime for the OBC state machine, FDIR, and swath triggers."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from GenADCS.backends.python.hardware.ADCS_subsystem.goals import CoordinateGoal, Goal
from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import (
    ADCSSMRuntime,
    GoalTable,
)
from GenADCS.backends.python.helpers.math.frames.conversions import ecef_to_gcrf_matrix

from .fdir import AngularRateMonitor
from .obc_subsystem import OBCSubsystem
from .state_machine import After, Completion, FDIRTriggered, InFOV, Mode, Telecommand

# FDIR monitor kinds.
_FDIR_ANGULAR_RATE = 1

# Earth rotation rate (rad/s) for conservative swath wake-up skipping.
_OMEGA_EARTH = 7.2921159e-5


class OBCRuntime(NamedTuple):
    """Flat numeric representation of OBC state machine, FDIR, and triggers."""

    enabled: bool
    initial_state: int
    n_states: int
    # CSR-indexed telecommand transitions per state:
    tc_trans_start: np.ndarray
    tc_trans_end: np.ndarray
    tc_trans_code: np.ndarray
    tc_trans_target: np.ndarray
    tc_trans_goal: np.ndarray
    # CSR-indexed orbital/nadir swath (InFOV) transitions per state:
    fov_trans_start: np.ndarray
    fov_trans_end: np.ndarray
    fov_ecef: np.ndarray
    fov_cos_limit: np.ndarray
    fov_negate: np.ndarray
    fov_target: np.ndarray
    fov_goal: np.ndarray
    # Auto-completion / timed transitions per state:
    state_duration: np.ndarray
    state_auto_target: np.ndarray
    # ADCS bindings per OBC state (-1 means no override):
    state_adcs_mode: np.ndarray
    state_adcs_goal: np.ndarray
    # Sorted telecommand timeline (terminated by inf / -1):
    tc_times: np.ndarray
    tc_codes: np.ndarray
    tc_goals: np.ndarray
    # FDIR configuration:
    fdir_enabled: bool
    fdir_safe_state: int
    fdir_monitor_kinds: np.ndarray
    fdir_monitor_thresholds: np.ndarray
    fdir_monitor_targets: np.ndarray
    # Human-readable state names (Python-side metadata):
    mode_names: tuple[str, ...]


def _resolve_target(name: str, modes_by_name: dict[str, Mode], leaf_ids: dict[str, int]) -> int:
    if name in leaf_ids:
        return leaf_ids[name]
    if name in modes_by_name and modes_by_name[name].submodes:
        sub = modes_by_name[name].initial_submode
        assert sub is not None
        return leaf_ids[f"{name}.{sub}"]
    raise ValueError(f"unknown target mode {name!r}")


def _resolve_sources(
    pattern: str,
    modes_by_name: dict[str, Mode],
    leaf_names: list[str],
    leaf_ids: dict[str, int],
) -> list[int]:
    if pattern == "*":
        return list(range(len(leaf_names)))
    if pattern.endswith(".*"):
        prefix = pattern[:-1]
        matched = [leaf_ids[n] for n in leaf_names if n.startswith(prefix)]
        if not matched:
            raise ValueError(f"transition source {pattern!r} matched no modes")
        return matched
    if pattern in leaf_ids:
        return [leaf_ids[pattern]]
    if pattern in modes_by_name and modes_by_name[pattern].submodes:
        prefix = f"{pattern}."
        return [leaf_ids[n] for n in leaf_names if n.startswith(prefix)]
    raise ValueError(f"unknown transition source {pattern!r}")


def _disabled_runtime() -> OBCRuntime:
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    return OBCRuntime(
        enabled=False,
        initial_state=0,
        n_states=0,
        tc_trans_start=empty_i,
        tc_trans_end=empty_i,
        tc_trans_code=empty_i,
        tc_trans_target=empty_i,
        tc_trans_goal=empty_i,
        fov_trans_start=empty_i,
        fov_trans_end=empty_i,
        fov_ecef=np.zeros((0, 3), dtype=np.float64),
        fov_cos_limit=empty_f,
        fov_negate=np.zeros(0, dtype=np.bool_),
        fov_target=empty_i,
        fov_goal=empty_i,
        state_duration=empty_f,
        state_auto_target=empty_i,
        state_adcs_mode=empty_i,
        state_adcs_goal=empty_i,
        tc_times=np.array([np.inf], dtype=np.float64),
        tc_codes=np.array([-1], dtype=np.int64),
        tc_goals=np.array([-1], dtype=np.int64),
        fdir_enabled=False,
        fdir_safe_state=0,
        fdir_monitor_kinds=empty_i,
        fdir_monitor_thresholds=empty_f,
        fdir_monitor_targets=empty_i,
        mode_names=(),
    )


def prepare_obc(
    obc: OBCSubsystem | None,
    adcs_sm_runtime: ADCSSMRuntime | None = None,
    goal_table: GoalTable | None = None,
) -> OBCRuntime:
    """Flatten OBC state machine, FDIR monitors, swath triggers, and timeline."""
    if obc is None or obc.state_machine is None:
        return _disabled_runtime()

    if goal_table is None:
        goal_table = GoalTable()
    adcs_mode_to_id = (
        {name: idx for idx, name in enumerate(adcs_sm_runtime.mode_names)}
        if adcs_sm_runtime is not None and adcs_sm_runtime.enabled
        else {}
    )

    sm = obc.state_machine
    modes_by_name = {mode.name: mode for mode in sm.modes}
    leaf_names: list[str] = []
    leaf_modes: list[Mode] = []
    leaf_parents: list[Mode | None] = []
    for mode in sm.modes:
        if mode.submodes:
            for sub in mode.submodes:
                leaf_names.append(f"{mode.name}.{sub.name}")
                leaf_modes.append(sub)
                leaf_parents.append(mode)
        else:
            leaf_names.append(mode.name)
            leaf_modes.append(mode)
            leaf_parents.append(None)

    leaf_ids = {name: idx for idx, name in enumerate(leaf_names)}
    n_states = len(leaf_names)
    initial_state = _resolve_target(sm.initial, modes_by_name, leaf_ids)

    cmd_to_code: dict[str, int] = {}

    def get_cmd_code(cmd: str) -> int:
        code = cmd_to_code.get(cmd)
        if code is None:
            code = len(cmd_to_code)
            cmd_to_code[cmd] = code
        return code

    state_duration = np.full(n_states, -1.0, dtype=np.float64)
    state_auto_target = np.full(n_states, -1, dtype=np.int64)
    state_adcs_mode = np.full(n_states, -1, dtype=np.int64)
    state_adcs_goal = np.full(n_states, -1, dtype=np.int64)

    for idx, (leaf, parent) in enumerate(zip(leaf_modes, leaf_parents)):
        if leaf.duration is not None and leaf.auto_target is not None:
            target_name = leaf.auto_target
            if parent is not None and "." not in target_name and f"{parent.name}.{target_name}" in leaf_ids:
                target_name = f"{parent.name}.{target_name}"
            state_duration[idx] = float(leaf.duration)
            state_auto_target[idx] = _resolve_target(target_name, modes_by_name, leaf_ids)

        binding = leaf.bindings.get("adcs")
        if binding is None and parent is not None:
            binding = parent.bindings.get("adcs")
        if isinstance(binding, str):
            if binding not in adcs_mode_to_id:
                raise ValueError(f"OBC binding references unknown ADCS mode {binding!r}")
            state_adcs_mode[idx] = adcs_mode_to_id[binding]
        elif isinstance(binding, Goal):
            state_adcs_goal[idx] = goal_table.register(binding)
        elif isinstance(binding, tuple) and len(binding) == 2:
            m_name, g_obj = binding
            if m_name not in adcs_mode_to_id:
                raise ValueError(f"OBC binding references unknown ADCS mode {m_name!r}")
            state_adcs_mode[idx] = adcs_mode_to_id[m_name]
            state_adcs_goal[idx] = goal_table.register(g_obj)

    # Collect per-state telecommand and InFOV transitions.
    per_state_tc: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_states)]
    per_state_fov: list[list[tuple[int, np.ndarray, float, bool, int, int]]] = [[] for _ in range(n_states)]

    for tr in sm.transitions:
        sources = _resolve_sources(tr.source, modes_by_name, leaf_names, leaf_ids)
        target_id = _resolve_target(tr.target, modes_by_name, leaf_ids)
        if isinstance(tr.trigger, Telecommand):
            code = get_cmd_code(tr.trigger.command)
            g_idx = goal_table.register(tr.trigger.goal) if tr.trigger.goal is not None else -1
            for src in sources:
                per_state_tc[src].append((tr.priority, code, target_id, g_idx))
        elif isinstance(tr.trigger, InFOV):
            coord = CoordinateGoal(tr.trigger.lat, tr.trigger.lon, tr.trigger.alt)
            cos_limit = math.cos(math.radians(tr.trigger.max_off_nadir_deg))
            if tr.trigger.goal is not None:
                g_idx = goal_table.register(tr.trigger.goal)
            elif not tr.trigger.negate:
                g_idx = goal_table.register(coord)
            else:
                g_idx = -1
            for src in sources:
                per_state_fov[src].append(
                    (tr.priority, coord.target_ecef_m, cos_limit, tr.trigger.negate, target_id, g_idx)
                )
        elif isinstance(tr.trigger, (Completion, After)):
            dur = float(tr.trigger.duration) if isinstance(tr.trigger, After) else None
            for src in sources:
                if dur is not None:
                    state_duration[src] = dur
                elif state_duration[src] <= 0.0:
                    raise ValueError(
                        f"Completion transition from {leaf_names[src]!r} requires Mode.duration"
                    )
                state_auto_target[src] = target_id
        elif isinstance(tr.trigger, FDIRTriggered):
            pass

    tc_trans_start = np.empty(n_states, dtype=np.int64)
    tc_trans_end = np.empty(n_states, dtype=np.int64)
    flat_codes: list[int] = []
    flat_targets: list[int] = []
    flat_tc_goals: list[int] = []
    cursor = 0
    for s in range(n_states):
        tc_trans_start[s] = cursor
        for _, code, target_id, g_idx in sorted(per_state_tc[s], key=lambda item: -item[0]):
            flat_codes.append(code)
            flat_targets.append(target_id)
            flat_tc_goals.append(g_idx)
            cursor += 1
        tc_trans_end[s] = cursor

    fov_trans_start = np.empty(n_states, dtype=np.int64)
    fov_trans_end = np.empty(n_states, dtype=np.int64)
    flat_ecef: list[np.ndarray] = []
    flat_cos: list[float] = []
    flat_neg: list[bool] = []
    flat_fov_targets: list[int] = []
    flat_fov_goals: list[int] = []
    cursor = 0
    for s in range(n_states):
        fov_trans_start[s] = cursor
        for _, ecef, cos_lim, neg, target_id, g_idx in sorted(per_state_fov[s], key=lambda item: -item[0]):
            flat_ecef.append(ecef)
            flat_cos.append(cos_lim)
            flat_neg.append(neg)
            flat_fov_targets.append(target_id)
            flat_fov_goals.append(g_idx)
            cursor += 1
        fov_trans_end[s] = cursor

    fov_ecef_arr = np.vstack(flat_ecef) if flat_ecef else np.zeros((0, 3), dtype=np.float64)

    # Merge timelines from OBCSubsystem and StateMachine, sorted by time.
    combined_timeline = list(sm.timeline) + list(obc.timeline)
    combined_timeline.sort(key=lambda item: item[0])
    tc_times = np.empty(len(combined_timeline) + 1, dtype=np.float64)
    tc_codes = np.empty(len(combined_timeline) + 1, dtype=np.int64)
    tc_goals = np.empty(len(combined_timeline) + 1, dtype=np.int64)
    for i, entry in enumerate(combined_timeline):
        tc_times[i] = float(entry[0])
        tc_codes[i] = get_cmd_code(entry[1])
        tc_goals[i] = goal_table.register(entry[2]) if len(entry) == 3 and entry[2] is not None else -1
    tc_times[-1] = np.inf
    tc_codes[-1] = -1
    tc_goals[-1] = -1

    fdir = obc.fdir
    if fdir is not None and fdir.monitors:
        fdir_enabled = True
        fdir_safe_state = _resolve_target(fdir.safe_mode, modes_by_name, leaf_ids)
        kinds = np.empty(len(fdir.monitors), dtype=np.int64)
        thresholds = np.empty(len(fdir.monitors), dtype=np.float64)
        targets = np.empty(len(fdir.monitors), dtype=np.int64)
        for i, mon in enumerate(fdir.monitors):
            if isinstance(mon, AngularRateMonitor):
                kinds[i] = _FDIR_ANGULAR_RATE
                thresholds[i] = float(mon.max_rate)
            else:
                raise TypeError(f"unsupported FDIR monitor: {type(mon).__name__}")
            targets[i] = (
                _resolve_target(mon.target_mode, modes_by_name, leaf_ids)
                if mon.target_mode is not None
                else fdir_safe_state
            )
    else:
        fdir_enabled = False
        fdir_safe_state = 0
        kinds = np.zeros(0, dtype=np.int64)
        thresholds = np.zeros(0, dtype=np.float64)
        targets = np.zeros(0, dtype=np.int64)

    return OBCRuntime(
        enabled=True,
        initial_state=initial_state,
        n_states=n_states,
        tc_trans_start=tc_trans_start,
        tc_trans_end=tc_trans_end,
        tc_trans_code=np.asarray(flat_codes, dtype=np.int64),
        tc_trans_target=np.asarray(flat_targets, dtype=np.int64),
        tc_trans_goal=np.asarray(flat_tc_goals, dtype=np.int64),
        fov_trans_start=fov_trans_start,
        fov_trans_end=fov_trans_end,
        fov_ecef=fov_ecef_arr,
        fov_cos_limit=np.asarray(flat_cos, dtype=np.float64),
        fov_negate=np.asarray(flat_neg, dtype=np.bool_),
        fov_target=np.asarray(flat_fov_targets, dtype=np.int64),
        fov_goal=np.asarray(flat_fov_goals, dtype=np.int64),
        state_duration=state_duration,
        state_auto_target=state_auto_target,
        state_adcs_mode=state_adcs_mode,
        state_adcs_goal=state_adcs_goal,
        tc_times=tc_times,
        tc_codes=tc_codes,
        tc_goals=tc_goals,
        fdir_enabled=fdir_enabled,
        fdir_safe_state=fdir_safe_state,
        fdir_monitor_kinds=kinds,
        fdir_monitor_thresholds=thresholds,
        fdir_monitor_targets=targets,
        mode_names=tuple(leaf_names),
    )


def init_obc_work(runtime: OBCRuntime) -> np.ndarray:
    """Allocate mutable scalar state: ``[state, tc_idx, auto_t, fdir_latched, next_fov_t]``."""
    work = np.empty(5, dtype=np.float64)
    work[0] = float(runtime.initial_state)
    work[1] = 0.0
    if runtime.enabled and runtime.state_duration[runtime.initial_state] > 0.0:
        work[2] = runtime.state_duration[runtime.initial_state]
    else:
        work[2] = np.inf
    work[3] = 0.0
    work[4] = 0.0
    return work


@njit(cache=True, inline="always")
def _enter_state(
    target: int,
    elapsed_s: float,
    obc: OBCRuntime,
    work: np.ndarray,
    goal_override: int,
) -> tuple[int, int, int]:
    work[0] = float(target)
    dur = obc.state_duration[target]
    work[2] = elapsed_s + dur if dur > 0.0 else np.inf
    work[4] = elapsed_s  # Allow immediate FOV evaluation in newly entered state.
    adcs_m = obc.state_adcs_mode[target]
    adcs_g = goal_override if goal_override >= 0 else obc.state_adcs_goal[target]
    return target, adcs_m, adcs_g


@njit(cache=True)
def step_obc(
    y: np.ndarray,
    elapsed_s: float,
    epoch_jd: float,
    dt_obc: float,
    obc: OBCRuntime,
    work: np.ndarray,
) -> tuple[int, int, int]:
    """Advance OBC FDIR, telecommands, auto-completion, and orbital swath triggers."""
    state = int(work[0])
    cmd_mode = -1
    cmd_goal = -1

    # 1. Fast FDIR check.
    if obc.fdir_enabled and work[3] == 0.0:
        for i in range(obc.fdir_monitor_kinds.size):
            if obc.fdir_monitor_kinds[i] == _FDIR_ANGULAR_RATE:
                wx, wy, wz = y[4], y[5], y[6]
                limit = obc.fdir_monitor_thresholds[i]
                if wx * wx + wy * wy + wz * wz > limit * limit:
                    work[3] = 1.0
                    return _enter_state(obc.fdir_monitor_targets[i], elapsed_s, obc, work, -1)

    # 2. Telecommand & auto-completion wake-up check.
    tc_idx = int(work[1])
    next_tc_t = obc.tc_times[tc_idx]
    auto_t = work[2]
    if elapsed_s >= next_tc_t or elapsed_s >= auto_t:
        while elapsed_s >= obc.tc_times[tc_idx]:
            cmd = obc.tc_codes[tc_idx]
            tl_goal = obc.tc_goals[tc_idx]
            tc_idx += 1
            work[1] = float(tc_idx)
            start = obc.tc_trans_start[state]
            end = obc.tc_trans_end[state]
            for k in range(start, end):
                if obc.tc_trans_code[k] == cmd:
                    target = obc.tc_trans_target[k]
                    if state == obc.fdir_safe_state and target != obc.fdir_safe_state:
                        work[3] = 0.0
                    tr_goal = obc.tc_trans_goal[k]
                    override = tl_goal if tl_goal >= 0 else tr_goal
                    state, cmd_mode, cmd_goal = _enter_state(target, elapsed_s, obc, work, override)
                    break

        if elapsed_s >= work[2]:
            auto_target = obc.state_auto_target[state]
            if auto_target >= 0:
                state, cm, cg = _enter_state(auto_target, elapsed_s, obc, work, -1)
                if cm >= 0:
                    cmd_mode = cm
                if cg >= 0:
                    cmd_goal = cg

    # 3. Active-state orbital/nadir swath (InFOV) check with spatial skip timer.
    fov_start = obc.fov_trans_start[state]
    fov_end = obc.fov_trans_end[state]
    if fov_start < fov_end and elapsed_s >= work[4]:
        r_sat = y[7:10]
        v_sat = y[10:13]
        r_norm = np.sqrt(r_sat[0] * r_sat[0] + r_sat[1] * r_sat[1] + r_sat[2] * r_sat[2])
        v_norm = np.sqrt(v_sat[0] * v_sat[0] + v_sat[1] * v_sat[1] + v_sat[2] * v_sat[2])
        rot = ecef_to_gcrf_matrix(epoch_jd, epoch_jd)
        min_skip = np.inf

        for k in range(fov_start, fov_end):
            t_ecef = obc.fov_ecef[k]
            t_eci = rot @ t_ecef
            dx = t_eci[0] - r_sat[0]
            dy = t_eci[1] - r_sat[1]
            dz = t_eci[2] - r_sat[2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            rt_sq = t_ecef[0] * t_ecef[0] + t_ecef[1] * t_ecef[1] + t_ecef[2] * t_ecef[2]
            r_horizon = np.sqrt(max(0.0, r_norm * r_norm - rt_sq))

            if dist > r_horizon:
                in_cone = False
                v_max = v_norm + _OMEGA_EARTH * np.sqrt(rt_sq)
                dt_skip = (dist - r_horizon) / max(1.0, v_max)
                if 0.9 * dt_skip < min_skip:
                    min_skip = 0.9 * dt_skip
            else:
                # Off-nadir cosine: dot(rel / dist, -r_sat / r_norm)
                cos_off_nadir = -(dx * r_sat[0] + dy * r_sat[1] + dz * r_sat[2]) / (dist * r_norm)
                in_cone = cos_off_nadir >= obc.fov_cos_limit[k]
                min_skip = 0.0

            triggered = (not in_cone) if obc.fov_negate[k] else in_cone
            if triggered:
                state, cm, cg = _enter_state(
                    obc.fov_target[k], elapsed_s, obc, work, obc.fov_goal[k]
                )
                if cm >= 0:
                    cmd_mode = cm
                if cg >= 0:
                    cmd_goal = cg
                return state, cmd_mode, cmd_goal

        if min_skip > dt_obc and not np.isinf(min_skip):
            work[4] = elapsed_s + min_skip
        else:
            work[4] = elapsed_s + dt_obc

    return state, cmd_mode, cmd_goal
