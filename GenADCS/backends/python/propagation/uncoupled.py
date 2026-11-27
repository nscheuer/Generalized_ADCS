"""Numba coordinator for uncoupled Keplerian orbit and attitude propagation."""

import math
from threading import Event, Thread

import numpy as np
from numba import njit

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.runtime import (
    prepare_estimator,
    update_estimator,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.controller import (
    CTRL_ATTITUDE,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.runtime import (
    ControllerRuntime,
    ControllerTable,
    init_controller_work,
    prepare_controllers,
    step_controller,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    SensorConfig,
    prepare_sensors,
    sample_sensors,
    attitude_references,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import (
    ADCSSMRuntime,
    GoalTable,
    evaluate_goal,
    init_adcs_work,
    prepare_adcs_sm,
    step_adcs_sm,
)
from GenADCS.backends.python.hardware.OBC_subsystem.runtime import (
    OBCRuntime,
    init_obc_work,
    prepare_obc,
    step_obc,
)
from GenADCS.backends.python.propagation.attitude_propagators.rk4_attitude_propagator import (
    _rk4_attitude_step,
)
from GenADCS.backends.python.propagation.coupled_propagators._coupled_dynamics import (
    SECONDS_PER_DAY,
    RuntimeParameters,
    prepare_runtime,
    project_wheel_momentum,
)
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import (
    pack_state,
)
from GenADCS.backends.python.propagation.orbital_propagators.keplerian_orbital_propagator import (
    keplerian_step,
)
from GenADCS.backends.python.state import State
from GenADCS.backends.python.propagation.attitude_propagators.reference_attitude_propagator import set_reference_attitude


@njit(cache=True, nogil=True)
def _simulate_uncoupled_loop(
    y: np.ndarray,
    initial_jd: float,
    dt: float,
    full_steps: int,
    remainder: float,
    runtime: RuntimeParameters,
    output: np.ndarray,
    record_every: int,
    sensor_every: int,
    sensor_config: SensorConfig,
    clean: np.ndarray,
    reading: np.ndarray,
    seed: int,
    estimator_kind: int,
    estimate_q: np.ndarray,
    estimate_w: np.ndarray,
    estimate_p: np.ndarray,
    acceleration_psd: np.ndarray,
    estimates: np.ndarray,
    estimator_flags: int,
    completed_steps: np.ndarray,
    progress_every: int,
    reference_mode: int,
    obc_every: int,
    obc_runtime: OBCRuntime,
    obc_work: np.ndarray,
    adcs_sm_every: int,
    adcs_sm_runtime: ADCSSMRuntime,
    adcs_work: np.ndarray,
    sm_trace: np.ndarray,
    ctrl_rt: ControllerRuntime,
    ctrl_work: np.ndarray,
    default_ctrl_idx: int,
    ctrl_log_flags: int,
    ctrl_log: np.ndarray,
    cmd_wheel_log: np.ndarray,
    cmd_dipole_log: np.ndarray,
) -> None:
    total_steps = full_steps + int(remainder > 0.0)
    active_obc_mode = int(obc_work[0])
    active_adcs_mode = int(adcs_work[0])
    active_adcs_goal = int(adcs_work[1])
    prev_ctrl_idx = -1
    pending_cmd_mode = -1
    pending_cmd_goal = -1
    if obc_every > 0:
        init_state = int(obc_work[0])
        pending_cmd_mode = obc_runtime.state_adcs_mode[init_state]
        pending_cmd_goal = obc_runtime.state_adcs_goal[init_state]
        active_obc_mode, cm, cg = step_obc(
            y, 0.0, initial_jd, dt * obc_every, obc_runtime, obc_work
        )
        if cm >= 0:
            pending_cmd_mode = cm
        if cg >= 0:
            pending_cmd_goal = cg
        sm_trace[0, 0] = active_obc_mode
    if adcs_sm_every > 0:
        active_adcs_mode, active_adcs_goal = step_adcs_sm(
            0.0, adcs_sm_runtime, adcs_work, pending_cmd_mode, pending_cmd_goal
        )
        pending_cmd_mode = -1
        pending_cmd_goal = -1
        sm_trace[0, 1] = active_adcs_mode
        sm_trace[0, 2] = active_adcs_goal
    if ctrl_log_flags:
        if ctrl_log_flags & 1:
            gq, gw = evaluate_goal(
                active_adcs_goal,
                adcs_sm_runtime.goal_kinds,
                adcs_sm_runtime.goal_params,
                y,
                initial_jd,
                runtime,
            )
            ctrl_log[0, 0:4] = gq
            ctrl_log[0, 4:7] = gw
        if (ctrl_log_flags & 32) and runtime.wheel_commands.size > 0:
            cmd_wheel_log[0, :] = runtime.wheel_commands
        if (ctrl_log_flags & 64) and runtime.dipole_commands.size > 0:
            cmd_dipole_log[0, :] = runtime.dipole_commands

    last_ff_torque = np.full(3, np.nan, dtype=np.float64)
    last_ff_q = np.full(4, np.nan, dtype=np.float64)
    last_ff_w = np.full(3, np.nan, dtype=np.float64)
    last_ref_torque = np.full(3, np.nan, dtype=np.float64)
    if sensor_every:
        sample_sensors(
            y,
            initial_jd,
            runtime,
            sensor_config,
            clean[0],
            reading[0],
            seed,
            0,
        )
        if estimator_kind:
            update_estimator(
                estimator_kind,
                estimate_q,
                estimate_w,
                estimate_p,
                acceleration_psd,
                0.0,
                reading[0],
                sensor_config,
                estimator_flags,
                y[:4],
                y[4:7],
                estimates[0, 43:],
                references=attitude_references(y[7:10], initial_jd, runtime, sensor_config),
            )
            if estimator_flags & 1:
                estimates[0, :4] = estimate_q
            if estimator_flags & 2:
                estimates[0, 4:7] = estimate_w
            if estimator_flags & 1024:
                if estimator_kind == 2:
                    estimates[0, 7:56] = estimate_p.ravel()
                else:
                    estimates[0, 7:43] = estimate_p.ravel()

    for row in range(total_steps):
        elapsed = row * dt
        step_dt = remainder if row == full_steps else dt
        epoch_jd = initial_jd + elapsed / SECONDS_PER_DAY
        # Attitude sees the orbit at the start of this step. The orbit then
        # advances using its independent Keplerian two-body model.
        if reference_mode < 0:
            y = _rk4_attitude_step(y, epoch_jd, step_dt, runtime)
        position, velocity = keplerian_step(y[7:10], y[10:13], runtime.mu, step_dt)
        y[7:10] = position
        y[10:13] = velocity
        if reference_mode >= 0:
            set_reference_attitude(y, epoch_jd + step_dt / SECONDS_PER_DAY, reference_mode, runtime)
        completed = row + 1
        if obc_every > 0 and completed <= full_steps and completed % obc_every == 0:
            elapsed_now = completed * dt
            active_obc_mode, cm, cg = step_obc(
                y,
                elapsed_now,
                initial_jd + elapsed_now / SECONDS_PER_DAY,
                dt * obc_every,
                obc_runtime,
                obc_work,
            )
            if cm >= 0:
                pending_cmd_mode = cm
            if cg >= 0:
                pending_cmd_goal = cg
        if adcs_sm_every > 0 and completed <= full_steps and completed % adcs_sm_every == 0:
            active_adcs_mode, active_adcs_goal = step_adcs_sm(
                completed * dt,
                adcs_sm_runtime,
                adcs_work,
                pending_cmd_mode,
                pending_cmd_goal,
            )
            pending_cmd_mode = -1
            pending_cmd_goal = -1
        if (
            sensor_every > 0
            and completed <= full_steps
            and completed % sensor_every == 0
        ):
            index = completed // sensor_every
            sample_sensors(
                y,
                initial_jd + completed * dt / SECONDS_PER_DAY,
                runtime,
                sensor_config,
                clean[index],
                reading[index],
                seed,
                index,
            )
            if estimator_kind:
                update_estimator(
                    estimator_kind,
                    estimate_q,
                    estimate_w,
                    estimate_p,
                    acceleration_psd,
                    dt * sensor_every,
                    reading[index],
                    sensor_config,
                    estimator_flags,
                    y[:4],
                    y[4:7],
                    estimates[index, 43:],
                    references=attitude_references(y[7:10], initial_jd + completed * dt / SECONDS_PER_DAY, runtime, sensor_config),
                )
                if estimator_flags & 1:
                    estimates[index, :4] = estimate_q
                if estimator_flags & 2:
                    estimates[index, 4:7] = estimate_w
                if estimator_flags & 1024:
                    if estimator_kind == 2:
                        estimates[index, 7:56] = estimate_p.ravel()
                    else:
                        estimates[index, 7:43] = estimate_p.ravel()
            if ctrl_rt.enabled:
                ctrl_idx = (
                    adcs_sm_runtime.mode_default_controller[active_adcs_mode]
                    if adcs_sm_runtime.enabled
                    else default_ctrl_idx
                )
                if ctrl_idx != prev_ctrl_idx:
                    ctrl_work[:] = 0.0
                    prev_ctrl_idx = ctrl_idx
                cur_jd_ctrl = initial_jd + completed * dt / SECONDS_PER_DAY
                if ctrl_rt.kinds[ctrl_idx] == CTRL_ATTITUDE:
                    q_g, w_g = evaluate_goal(
                        active_adcs_goal,
                        adcs_sm_runtime.goal_kinds,
                        adcs_sm_runtime.goal_params,
                        y,
                        cur_jd_ctrl,
                        runtime,
                    )
                else:
                    q_g = np.full(4, np.nan, dtype=np.float64)
                    w_g = np.full(3, np.nan, dtype=np.float64)
                ctrl_q = estimate_q if estimator_kind else y[0:4]
                ctrl_w = estimate_w if estimator_kind else y[4:7]
                last_ff_torque, last_ff_q, last_ff_w, last_ref_torque = step_controller(
                    ctrl_idx,
                    dt * sensor_every,
                    ctrl_q,
                    ctrl_w,
                    q_g,
                    w_g,
                    y[13:],
                    reading[index],
                    sensor_config,
                    ctrl_rt,
                    ctrl_work,
                    runtime.wheel_commands,
                    runtime.dipole_commands,
                )
        if completed % record_every == 0:
            rec_idx = completed // record_every
            output[rec_idx] = y
            if obc_every > 0:
                sm_trace[rec_idx, 0] = active_obc_mode
            if adcs_sm_every > 0:
                sm_trace[rec_idx, 1] = active_adcs_mode
                sm_trace[rec_idx, 2] = active_adcs_goal
            if ctrl_log_flags:
                if ctrl_log_flags & 1:
                    cur_jd = initial_jd + min(completed * dt, full_steps * dt + remainder) / SECONDS_PER_DAY
                    gq, gw = evaluate_goal(
                        active_adcs_goal,
                        adcs_sm_runtime.goal_kinds,
                        adcs_sm_runtime.goal_params,
                        y,
                        cur_jd,
                        runtime,
                    )
                    ctrl_log[rec_idx, 0:4] = gq
                    ctrl_log[rec_idx, 4:7] = gw
                if ctrl_log_flags & 2:
                    ctrl_log[rec_idx, 7:10] = last_ff_torque
                if ctrl_log_flags & 4:
                    ctrl_log[rec_idx, 10:14] = last_ff_q
                if ctrl_log_flags & 8:
                    ctrl_log[rec_idx, 14:17] = last_ff_w
                if ctrl_log_flags & 16:
                    ctrl_log[rec_idx, 17:20] = last_ref_torque
                if (ctrl_log_flags & 32) and runtime.wheel_commands.size > 0:
                    cmd_wheel_log[rec_idx, :] = runtime.wheel_commands
                if (ctrl_log_flags & 64) and runtime.dipole_commands.size > 0:
                    cmd_dipole_log[rec_idx, :] = runtime.dipole_commands
        elif completed == total_steps:
            output[-1] = y
            if obc_every > 0:
                sm_trace[-1, 0] = active_obc_mode
            if adcs_sm_every > 0:
                sm_trace[-1, 1] = active_adcs_mode
                sm_trace[-1, 2] = active_adcs_goal
            if ctrl_log_flags:
                if ctrl_log_flags & 1:
                    cur_jd = initial_jd + (full_steps * dt + remainder) / SECONDS_PER_DAY
                    gq, gw = evaluate_goal(
                        active_adcs_goal,
                        adcs_sm_runtime.goal_kinds,
                        adcs_sm_runtime.goal_params,
                        y,
                        cur_jd,
                        runtime,
                    )
                    ctrl_log[-1, 0:4] = gq
                    ctrl_log[-1, 4:7] = gw
                if ctrl_log_flags & 2:
                    ctrl_log[-1, 7:10] = last_ff_torque
                if ctrl_log_flags & 4:
                    ctrl_log[-1, 10:14] = last_ff_q
                if ctrl_log_flags & 8:
                    ctrl_log[-1, 14:17] = last_ff_w
                if ctrl_log_flags & 16:
                    ctrl_log[-1, 17:20] = last_ref_torque
                if (ctrl_log_flags & 32) and runtime.wheel_commands.size > 0:
                    cmd_wheel_log[-1, :] = runtime.wheel_commands
                if (ctrl_log_flags & 64) and runtime.dipole_commands.size > 0:
                    cmd_dipole_log[-1, :] = runtime.dipole_commands
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == total_steps
        ):
            completed_steps[0] = completed


def simulate_uncoupled(
    state: State,
    satellite: Satellite,
    environment: Environment,
    dynamics: DynamicsConfig,
    dt: float,
    tf: float,
    record_every: int,
    sensor_every: int,
    seed: int,
    estimator_flags: int,
    progress: bool = False,
    progress_chunk_steps: int = 500,
    reference_mode: int = -1,
    ctrl_log_flags: int = 0,
) -> tuple[np.ndarray, ...]:
    """Prepare once, execute every step in one JIT call, return numeric samples."""
    tf = float(tf)
    if not math.isfinite(tf) or tf < 0:
        raise ValueError("tf must be finite and nonnegative seconds")
    if isinstance(record_every, bool) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if (
        isinstance(progress_chunk_steps, bool)
        or not isinstance(progress_chunk_steps, int)
        or progress_chunk_steps < 1
    ):
        raise ValueError("progress_chunk_steps must be a positive integer")
    runtime = prepare_runtime(satellite, environment, dynamics)
    y = pack_state(state, runtime.wheel_commands.size)
    if runtime.wheel_commands.size:
        project_wheel_momentum(y, runtime)
    full_steps = math.floor(tf / dt)
    remainder = tf - full_steps * dt
    if remainder <= 1e-12 * max(1.0, tf):
        remainder = 0.0
    total_steps = full_steps + int(remainder > 0.0)
    sampled_steps = np.arange(0, total_steps + 1, record_every, dtype=np.int64)
    elapsed = np.minimum(sampled_steps.astype(np.float64) * dt, tf)
    if sampled_steps[-1] != total_steps:
        elapsed = np.append(elapsed, tf)
    else:
        elapsed[-1] = tf
    output = np.empty((elapsed.size, y.size), dtype=np.float64)
    if reference_mode >= 0:
        set_reference_attitude(y, state.orbit.julian_date, reference_mode, runtime)
    output[0] = y
    sensor_config = prepare_sensors(
        satellite.adcs.sensors if sensor_every else [], environment
    )
    width = max(1, int(np.sum(sensor_config[-2])))
    sensor_rows = full_steps // sensor_every + 1 if sensor_every else 1
    clean = np.full((sensor_rows, width), np.nan)
    reading = np.full_like(clean, np.nan)
    estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd = (
        prepare_estimator(satellite.adcs.estimator)
    )
    estimates = np.full((sensor_rows, 96), np.nan)
    completed_steps = np.zeros(1, dtype=np.int64)
    goal_table = GoalTable()
    ctrl_table = ControllerTable()
    default_ctrl_idx = ctrl_table.register(satellite.adcs.controller)
    adcs_sm_runtime = prepare_adcs_sm(
        satellite.adcs.state_machine, goal_table, ctrl_table
    )
    wheel_torque_limits = np.asarray(
        [w.max_torque for w in satellite.adcs.reaction_wheels], dtype=np.float64
    )
    bus_inertia = np.linalg.inv(runtime.inverse_bus_inertia) if runtime.wheel_axes.shape[0] > 0 else runtime.inertia
    ctrl_rt = prepare_controllers(
        (),
        satellite.adcs.sensors,
        runtime.mtq_axes,
        runtime.mtq_max_dipole,
        ctrl_table,
        wheel_axes=runtime.wheel_axes,
        wheel_max_torque=wheel_torque_limits,
        total_inertia=runtime.inertia,
        bus_inertia=bus_inertia,
    )
    ctrl_work = init_controller_work()
    obc_runtime = prepare_obc(satellite.obc, adcs_sm_runtime, goal_table)
    obc_every = satellite.obc.N if obc_runtime.enabled and satellite.obc is not None else 0
    adcs_sm_every = satellite.adcs.N if (adcs_sm_runtime.enabled or obc_every > 0) else 0
    obc_work = init_obc_work(obc_runtime)
    adcs_work = init_adcs_work(adcs_sm_runtime)
    sm_trace = np.full((elapsed.size, 3), -1, dtype=np.int64)
    ctrl_log = (
        np.full((elapsed.size, 20), np.nan, dtype=np.float64)
        if (ctrl_log_flags & 31)
        else np.empty((0, 0), dtype=np.float64)
    )
    cmd_wheel_log = (
        np.full((elapsed.size, runtime.wheel_commands.size), np.nan, dtype=np.float64)
        if (ctrl_log_flags & 32)
        else np.empty((0, 0), dtype=np.float64)
    )
    cmd_dipole_log = (
        np.full((elapsed.size, runtime.dipole_commands.size), np.nan, dtype=np.float64)
        if (ctrl_log_flags & 64)
        else np.empty((0, 0), dtype=np.float64)
    )
    has_sm = obc_every > 0 or adcs_sm_every > 0
    args = (
        y,
        state.orbit.julian_date,
        dt,
        full_steps,
        remainder,
        runtime,
        output,
        record_every,
        sensor_every,
        sensor_config,
        clean,
        reading,
        seed,
        estimator_kind,
        estimate_q,
        estimate_w,
        estimate_p,
        acceleration_psd,
        estimates,
        estimator_flags,
        completed_steps,
    )
    if progress:
        from tqdm import tqdm

        finished = Event()

        def monitor() -> None:
            with tqdm(
                total=total_steps,
                desc="Running Keplerian + " + ("RK4" if reference_mode < 0 else "reference"),
                unit="step",
            ) as bar:
                while not finished.wait(0.1):
                    current = int(completed_steps[0])
                    if current > bar.n:
                        bar.update(current - bar.n)
                current = int(completed_steps[0])
                if current > bar.n:
                    bar.update(current - bar.n)

        thread = Thread(target=monitor, daemon=True)
        thread.start()
        try:
            _simulate_uncoupled_loop(
                *args, progress_chunk_steps, reference_mode,
                obc_every, obc_runtime, obc_work,
                adcs_sm_every, adcs_sm_runtime, adcs_work, sm_trace,
                ctrl_rt, ctrl_work, default_ctrl_idx,
                ctrl_log_flags, ctrl_log, cmd_wheel_log, cmd_dipole_log,
            )
        finally:
            finished.set()
            thread.join()
    else:
        _simulate_uncoupled_loop(
            *args, 0, reference_mode,
            obc_every, obc_runtime, obc_work,
            adcs_sm_every, adcs_sm_runtime, adcs_work, sm_trace,
            ctrl_rt, ctrl_work, default_ctrl_idx,
            ctrl_log_flags, ctrl_log, cmd_wheel_log, cmd_dipole_log,
        )
    if ctrl_log_flags > 0:
        return (
            (elapsed, output, clean, reading, estimates, sm_trace, ctrl_log, cmd_wheel_log, cmd_dipole_log)
            if estimator_kind
            else (elapsed, output, clean, reading, sm_trace, ctrl_log, cmd_wheel_log, cmd_dipole_log) if sensor_every
            else (elapsed, output, sm_trace, ctrl_log, cmd_wheel_log, cmd_dipole_log)
        )
    if has_sm:
        return (
            (elapsed, output, clean, reading, estimates, sm_trace)
            if estimator_kind
            else (elapsed, output, clean, reading, sm_trace) if sensor_every else (elapsed, output, sm_trace)
        )
    return (
        (elapsed, output, clean, reading, estimates)
        if estimator_kind
        else (elapsed, output, clean, reading) if sensor_every else (elapsed, output)
    )
