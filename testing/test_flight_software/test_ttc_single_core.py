import numpy as np

from ADCS.flight_software.single_core.ttc_single_core import TTC_Single_Core
from ADCS.flight_software.tasks.task import Task


def make_logger(memory, key, value_fn=None):
    def _task(t, mem):
        value = t if value_fn is None else value_fn(t, mem)
        mem[key].append(value)

    return _task


def make_task(name, callback, rate_hz, wcet, priority, enabled=True):
    return Task(
        name=name,
        callback=callback,
        rate_hz=rate_hz,
        wcet=wcet,
        priority=priority,
        enabled=enabled,
    )


def test_priority_ordering():
    memory = {"log": []}

    tasks = [
        make_task("low", lambda t, m: m["log"].append("LOW"), 1.0, 0.01, 2),
        make_task("high", lambda t, m: m["log"].append("HIGH"), 1.0, 0.01, 0),
    ]

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=tasks, memory=memory)
    core.step()

    assert memory["log"] == ["HIGH", "LOW"]


def test_equal_priority_preserves_insertion_order():
    memory = {"log": []}

    tasks = [
        make_task("first", lambda t, m: m["log"].append("FIRST"), 1.0, 0.01, 0),
        make_task("second", lambda t, m: m["log"].append("SECOND"), 1.0, 0.01, 0),
    ]

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=tasks, memory=memory)
    core.step()

    assert memory["log"] == ["FIRST", "SECOND"]


def test_frequency_correctness():
    memory = {"A": [], "B": []}

    tasks = [
        make_task("A", make_logger(memory, "A"), 5.0, 0.001, 0),
        make_task("B", make_logger(memory, "B"), 1.0, 0.001, 1),
    ]

    core = TTC_Single_Core(base_rate_hz=5.0, tasks=tasks, memory=memory)

    for _ in range(10):
        core.step()

    assert len(memory["A"]) == 10
    assert len(memory["B"]) == 2


def test_runs_at_boot():
    memory = {"log": []}
    task = make_task("slow", make_logger(memory, "log"), 1.0, 0.001, 0)
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[task], memory=memory)

    core.step()

    assert memory["log"] == [0.0]


def test_task_disable():
    memory = {"log": []}
    task = make_task("disabled", make_logger(memory, "log"), 1.0, 0.001, 0, enabled=False)

    core = TTC_Single_Core(base_rate_hz=1.0, tasks=[task], memory=memory)
    core.step()

    assert memory["log"] == []
    assert task.exec_count == 0


def test_cpu_budget_enforced():
    memory = {"log": []}

    tasks = [
        make_task("A", lambda t, m: m["log"].append("A"), 1.0, 0.08, 0),
        make_task("B", lambda t, m: m["log"].append("B"), 1.0, 0.05, 1),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)
    core.step()

    assert memory["log"] == ["A"]


def test_skipped_release_runs_on_next_frame():
    memory = {"log": []}
    high = make_task("high", lambda t, m: m["log"].append(("high", t)), 1.0, 0.08, 0)
    low = make_task("low", lambda t, m: m["log"].append(("low", t)), 1.0, 0.05, 1)
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[high, low], memory=memory)

    core.step()
    core.step()

    assert memory["log"] == [("high", 0.0), ("low", 0.1)]
    assert high.exec_count == 1
    assert low.exec_count == 1
    assert np.isclose(high.next_release_time, 1.0)
    assert np.isclose(low.next_release_time, 1.0)


def test_next_release_time_advances_only_when_task_runs():
    memory = {"log": []}
    task = make_task("task", lambda t, m: m["log"].append(t), 2.0, 0.05, 1)
    blocker = make_task("blocker", lambda t, m: None, 10.0, 0.08, 0)
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[blocker, task], memory=memory)

    core.step()

    assert task.exec_count == 0
    assert np.isclose(task.next_release_time, 0.0)


def test_trace_records_run_and_budget_miss():
    memory = {"log": []}
    tasks = [
        make_task("A", lambda t, m: m["log"].append("A"), 1.0, 0.08, 0),
        make_task("B", lambda t, m: m["log"].append("B"), 1.0, 0.05, 1),
    ]
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)

    core.step()

    assert len(core._trace) == 2

    ran_event, missed_event = core._trace
    assert ran_event.task == "A"
    assert ran_event.released is True
    assert ran_event.ran is True
    assert np.isclose(ran_event.charged_time, 0.08)
    assert ran_event.note == "OK"

    assert missed_event.task == "B"
    assert missed_event.released is True
    assert missed_event.ran is False
    assert np.isclose(missed_event.cpu_budget_before, 0.02)
    assert np.isclose(missed_event.cpu_budget_after, 0.02)
    assert missed_event.note == "MISS_NO_BUDGET"


def test_idle_task_only_runs_if_slack():
    memory = {"log": []}

    tasks = [
        make_task("real", lambda t, m: m["log"].append("REAL"), 1.0, 0.08, 0),
        make_task("idle", lambda t, m: m["log"].append("IDLE"), 10.0, 1.0, 9),
    ]

    core = TTC_Single_Core(base_rate_hz=10.0, tasks=tasks, memory=memory)
    core.step()

    assert "IDLE" not in memory["log"]


def test_frame_slack_recorded_for_empty_frame():
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[], memory={})

    core.step()

    assert core._frame_slack == [(0.0, 0.1)]


def test_latched_command_behavior():
    memory = {"u_cmd": 0}

    def controller(t, mem):
        mem["u_cmd"] += 1

    task = make_task("ctrl", controller, 1.0, 0.001, 0)
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[task], memory=memory)

    for _ in range(10):
        core.step()

    assert memory["u_cmd"] == 1


def test_estimator_faster_than_controller():
    memory = {"x_hat": 0, "u_cmd": None}

    def estimator(t, mem):
        mem["x_hat"] += 1

    def controller(t, mem):
        mem["u_cmd"] = mem["x_hat"]

    tasks = [
        make_task("est", estimator, 5.0, 0.001, 0),
        make_task("ctrl", controller, 1.0, 0.001, 1),
    ]

    core = TTC_Single_Core(base_rate_hz=5.0, tasks=tasks, memory=memory)

    for _ in range(5):
        core.step()

    assert memory["u_cmd"] == 1


def test_time_advances_only_by_base_dt():
    core = TTC_Single_Core(base_rate_hz=10.0, tasks=[], memory={})

    core.step()
    core.step()

    assert np.isclose(core.t_fsw, 0.2)


def test_background_task_slicing():
    memory = {"progress": 0}

    def planner(t, mem):
        mem["progress"] += 1

    planner_task = make_task("planner", planner, 1.0, 0.01, 5)
    core = TTC_Single_Core(base_rate_hz=1.0, tasks=[planner_task], memory=memory)

    for _ in range(5):
        core.step()

    assert memory["progress"] == 5
