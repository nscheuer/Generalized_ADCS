__all__ = ["Task"]

from typing import Callable


class Task:
    r"""
    Periodic task definition for a time-triggered single-core scheduler.

    This class models a **cooperative, non-preemptive, periodic task** intended
    to be executed by a time-triggered scheduler such as
    :class:`~ADCS.flight_software.schedulers.ttc_single_core.TTC_Single_Core`.

    Each task is characterized by:

    * A fixed execution period :math:`T`
    * A fixed worst-case execution time (WCET) :math:`C`
    * A static priority
    * A callback function representing the task body

    The execution period is derived from the configured rate:

    .. math::

        T = \frac{1}{f}

    where :math:`f` is the task rate in Hertz.

    Scheduling and execution assumptions:

    * Tasks are **non-preemptive**
    * If a task executes, it consumes exactly its WCET
    * If insufficient CPU budget exists, the task is skipped
    * WCET overruns are not modeled at this level
    * Task releases are consumed only upon execution

    This abstraction is designed for:

    * Deterministic schedulability analysis
    * TTC (Time-Triggered Cooperative) flight software
    * Offline simulation and testing

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
        r"""
        Initialize a periodic task definition.

        This constructor defines all static properties of the task. Dynamic
        scheduling state (such as next release time and execution count) is
        initialized to default values and maintained by the scheduler.

        The task execution period is computed as:

        .. math::

            T = \frac{1}{\text{rate\_hz}}

        The scheduler charges the WCET to the CPU budget **only if the task runs**.

        :param name:
            Human-readable task name.
        :type name: str

        :param callback:
            Function executed when the task runs.

            The callback must have the signature:

            .. code-block:: python

                callback(t_fsw: float, memory: dict) -> None

            where ``t_fsw`` is the logical flight software time and ``memory`` is the
            shared inter-task memory dictionary.
        :type callback: Callable

        :param rate_hz:
            Task execution rate in Hertz.
        :type rate_hz: float

        :param wcet:
            Worst-case execution time of the task, in seconds.
        :type wcet: float

        :param priority:
            Fixed scheduling priority.
            Lower numerical values indicate higher priority.
        :type priority: int

        :param enabled:
            Whether the task is initially enabled.
        :type enabled: bool

        :return:
            None
        :rtype:
            None

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
