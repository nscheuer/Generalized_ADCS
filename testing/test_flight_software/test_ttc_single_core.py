import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.flight_software.single_core.ttc_single_core import TTC_Single_Core
from ADCS.flight_software.tasks.task import Task


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_logger(memory, key):
    def _task(t, mem):
        mem[key].append(t)
    return _task


# ---------------------------------------------------------------------
# Tier 1: Scheduling correctness
# ---------------------------------------------------------------------

def test_priority_ordering():
    """
    Higher-priority tasks must always run before lower-priority ones.
    """
    memory = {"log": []}

    def high(t, mem): mem["log"].append("HIGH")
    def low(t, mem): mem["log"].append("LOW")

    tasks = [
        Task("low", low, rate_hz=1.0, wcet=0.01, priority=2),
        Task("high", high, rate_hz=1.0, wcet=0.01, priority=0),
    ]

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=tasks, memory=memory)
    core.step()

    assert memory["log"] == ["HIGH", "LOW"]


def test_frequency_correctness():
    """
    Tasks must run exactly at their configured rate.
    """
    memory = {"A": [], "B": []}

    tasks = [
        Task("A", make_logger(memory, "A"), rate_hz=5.0, wcet=0.001, priority=0),
        Task("B", make_logger(memory, "B"), rate_hz=1.0, wcet=0.001, priority=1),
    ]

    core = TTC_Single_Core(base_rate_hz=5.0, tasks=tasks, memory=memory)

    for _ in range(10):   # 2 seconds
        core.step()

    assert len(memory["A"]) == 10
    assert len(memory["B"]) == 2


def test_runs_at_boot():
    memory = {"log": []}

    tasks = [
        Task("slow", make_logger(memory, "log"), rate_hz=1.0, wcet=0.001, priority=0),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)
    core.step()

    assert len(memory["log"]) == 1


def test_task_disable():
    """
    Disabled tasks must not execute.
    """
    memory = {"log": []}

    task = Task("test", make_logger(memory, "log"), rate_hz=1.0, wcet=0.001, priority=0)
    task.enabled = False

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=[task], memory=memory)
    core.step()

    assert len(memory["log"]) == 0


# ---------------------------------------------------------------------
# Tier 2: CPU budget realism
# ---------------------------------------------------------------------

def test_cpu_budget_enforced():
    """
    Lower-priority tasks must be skipped if CPU budget is exhausted.
    """
    memory = {"log": []}

    tasks = [
        Task("A", lambda t, m: m["log"].append("A"),
             rate_hz=1.0, wcet=0.08, priority=0),
        Task("B", lambda t, m: m["log"].append("B"),
             rate_hz=1.0, wcet=0.05, priority=1),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)
    core.step()

    assert memory["log"] == ["A"]


def test_idle_task_only_runs_if_slack():
    """
    Idle must not preempt real work.
    """
    memory = {"log": []}

    tasks = [
        Task("real", lambda t, m: m["log"].append("REAL"),
             rate_hz=1.0, wcet=0.08, priority=0),
        Task("idle", lambda t, m: m["log"].append("IDLE"),
             rate_hz=10.0, wcet=1.0, priority=9),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)
    core.step()

    assert "IDLE" not in memory["log"]


# ---------------------------------------------------------------------
# Tier 3: Memory semantics
# ---------------------------------------------------------------------

def test_latched_command_behavior():
    """
    Controller must overwrite a single command register.
    """
    memory = {"u_cmd": 0}

    def controller(t, mem):
        mem["u_cmd"] += 1

    task = Task("ctrl", controller, rate_hz=1.0, wcet=0.001, priority=0)
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[task], memory=memory)

    for _ in range(10):
        core.step()

    assert memory["u_cmd"] == 1


def test_estimator_faster_than_controller():
    """
    Estimator may run faster, controller must see latest value only when it runs.
    """
    memory = {"x_hat": 0, "u_cmd": None}

    def estimator(t, mem):
        mem["x_hat"] += 1

    def controller(t, mem):
        mem["u_cmd"] = mem["x_hat"]

    tasks = [
        Task("est", estimator, rate_hz=5.0, wcet=0.001, priority=0),
        Task("ctrl", controller, rate_hz=1.0, wcet=0.001, priority=1),
    ]

    core = TTC_Single_Core(base_rate_hz=5.0, tasks=tasks, memory=memory)

    for _ in range(5):
        core.step()

    assert memory["u_cmd"] == 1


# ---------------------------------------------------------------------
# Tier 4: Time semantics
# ---------------------------------------------------------------------

def test_time_advances_only_by_base_dt():
    """
    Onboard time must advance deterministically.
    """
    memory = {}
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[], memory=memory)

    core.step()
    core.step()

    assert np.isclose(core.t_fsw, 0.2)


# ---------------------------------------------------------------------
# Tier 5: Planner slicing (generic)
# ---------------------------------------------------------------------

def test_background_task_slicing():
    """
    Planner-style task must advance incrementally.
    """
    memory = {"progress": 0}

    def planner(t, mem):
        mem["progress"] += 1

    planner_task = Task("planner", planner, rate_hz=1.0, wcet=0.01, priority=5)

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=[planner_task], memory=memory)

    for _ in range(5):
        core.step()

    assert memory["progress"] == 5

