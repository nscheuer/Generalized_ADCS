import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.flight_software.single_core.ttc_single_core import TTC_Single_Core
from ADCS.flight_software.tasks.task import Task

def os_task(t, mem):
    mem["log"].append(f"{t:.2f}: OS")


def estimator_task(t, mem):
    mem["log"].append(f"{t:.2f}: EST")


def controller_task(t, mem):
    mem["log"].append(f"{t:.2f}: CTRL")


def planner_task(t, mem):
    mem["planner_progress"] += 1
    mem["log"].append(f"{t:.2f}: PLAN (step {mem['planner_progress']})")



memory = {"log": [], "planner_progress": 0}

tasks = [
    Task("os", os_task, rate_hz=10.0, wcet=0.001, priority=0),
    Task("estimator", estimator_task, rate_hz=5.0, wcet=0.020, priority=1),
    Task("controller", controller_task, rate_hz=1.0, wcet=0.002, priority=2),

    # Slow background planner
    Task("planner", planner_task, rate_hz=0.5, wcet=0.010, priority=5),
]


core = TTC_Single_Core(
    base_rate_hz=10.0,  # 0.25 s frame
    tasks=tasks,
    memory=memory,
    debug=True,
)

# Run for 1 second (4 frames)
for _ in range(21):
    core.step()

print("\nExecution log:")
for line in memory["log"]:
    print(line)