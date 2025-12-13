
from typing import List

from ADCS.flight_software.tasks.task import Task

from typing import List, Callable

class TTC_Single_Core:
    def __init__(self, base_rate_hz: float, tasks: List[Task], memory: dict, debug: bool = False) -> None:
        self.base_rate_hz = base_rate_hz
        self.base_dt = 1.0 / base_rate_hz

        # Fixed-priority ordering (lower = higher priority)
        self.tasks = sorted(tasks, key=lambda t: t.priority)
        self.memory = memory
        self.debug = debug

        self.t_fsw = 0.0

    def step(self):
        """Advance the CPU by one base frame."""
        cpu_budget = self.base_dt

        if self.debug:
            print(f"\n[t = {self.t_fsw:.2f}s] CPU frame start")

        for task in self.tasks:
            if not task.enabled:
                continue

            if self.t_fsw + 1e-12 < task.next_release_time:
                continue

            if cpu_budget < task.wcet:
                if self.debug:
                    print(f"  MISS: {task.name} (no CPU budget)")
                continue

            if self.debug:
                print(f"  RUN: {task.name}")

            task.callback(self.t_fsw, self.memory)
            task.exec_count += 1
            task.next_release_time += task.period
            cpu_budget -= task.wcet

        self.t_fsw += self.base_dt
