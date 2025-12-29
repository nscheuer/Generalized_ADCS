__all__ = ["Task"]

from typing import Callable


class Task:
    """
    Periodic task definition for a time-triggered single-core scheduler.

    This class represents a cooperative, non-preemptive task that executes
    periodically according to a fixed rate and consumes a fixed amount of
    CPU time equal to its Worst-Case Execution Time (WCET).

    The scheduler assumes that:
    - If a task executes, it consumes exactly ``wcet`` seconds of CPU time
    - If insufficient budget is available, the task is skipped
    - Tasks do not overrun their WCET (overruns are modeled at a higher level)

    This deterministic model is suitable for:
    - Unit testing
    - Schedulability analysis
    - TTC flight software simulations
    """

    def __init__(
        self,
        name: str,
        callback: Callable,
        rate_hz: float,
        wcet: float,
        priority: int,
        enabled: bool = True,
    ) -> None:
        """
        Initialize a periodic task.

        Parameters
        ----------
        name : str
            Human-readable task name.

        callback : Callable
            Function executed when the task runs.
            Signature: ``callback(t_fsw: float, memory: dict)``.

        rate_hz : float
            Task execution rate in Hertz.

            The execution period is:

            .. math::

                T = \\frac{1}{\\text{rate\\_hz}}

        wcet : float
            Worst-case execution time (seconds).
            This value is *charged* to the CPU budget when the task runs.

        priority : int
            Fixed scheduling priority.
            Lower values indicate higher priority.

        enabled : bool, optional
            Whether the task is initially enabled.
        """
        self.name = name
        self.callback = callback

        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz

        self.wcet = wcet
        self.priority = priority
        self.enabled = enabled

        # Scheduler-maintained state
        self.next_release_time = 0.0
        self.exec_count = 0
