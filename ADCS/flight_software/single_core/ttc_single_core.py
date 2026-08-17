__all__ = ["TTC_Single_Core"]

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

from ADCS.flight_software.tasks.task import Task


@dataclass
class FrameEvent:
    r"""
    Record of a single task execution attempt within one scheduler base frame.

    This data structure represents **one accounting event** in the
    time-triggered scheduler. Each instance corresponds to a task that was
    *considered* during a specific frame, whether or not it actually ran.

    The record is **WCET-accounted**: if a task runs, it is charged exactly its
    configured WCET, independent of its internal execution behavior. This makes
    the scheduler deterministic and suitable for schedulability analysis.

    Conceptually, for a base frame of duration :math:`\Delta t`, the scheduler
    enforces the constraint:

    .. math::

        \sum_{i \in \text{ran}} C_i \le \Delta t

    where :math:`C_i` is the WCET of task *i*.

    :param t_fsw:
        Logical flight software time at the **start** of the frame, in seconds.
    :type t_fsw: float

    :param task:
        Name of the task associated with this event.
    :type task: str

    :param released:
        Indicates whether the task was eligible for execution
        (i.e. its release time had elapsed).
    :type released: bool

    :param ran:
        Indicates whether the task callback was actually executed.
    :type ran: bool

    :param charged_time:
        CPU time charged to the task during this frame, in seconds.
        For WCET accounting, this is either ``0`` or exactly ``wcet``.
    :type charged_time: float

    :param wcet:
        Worst-case execution time of the task, in seconds.
    :type wcet: float

    :param cpu_budget_before:
        Remaining CPU budget at the start of the task decision, in seconds.
    :type cpu_budget_before: float

    :param cpu_budget_after:
        Remaining CPU budget after the task decision, in seconds.
    :type cpu_budget_after: float

    :param note:
        Human-readable annotation describing the outcome
        (e.g. ``"OK"``, ``"MISS_NO_BUDGET"``).
    :type note: str

    """
    t_fsw: float
    task: str
    released: bool
    ran: bool
    charged_time: float
    wcet: float
    cpu_budget_before: float
    cpu_budget_after: float
    note: str


class TTC_Single_Core:
    r"""
    Single-core time-triggered cooperative scheduler with WCET-based accounting.

    This class models a **deterministic, non-preemptive, fixed-priority**
    scheduler intended to represent a spacecraft flight computer executing a
    static set of periodic tasks on a single CPU core.

    The scheduler operates on a fixed **base frame** of duration:

    .. math::

        \Delta t = \frac{1}{f_{\text{base}}}

    where :math:`f_{\text{base}}` is the base scheduler frequency.

    At the start of each frame:

    * The available CPU budget is reset to :math:`\Delta t`
    * Tasks are evaluated in **ascending priority order**
      (lower numerical value means higher priority)
    * A task may execute **at most once per release**
    * If a task executes, it consumes exactly its WCET from the frame budget

    The scheduler enforces the per-frame constraint:

    .. math::

        \sum_{i \in \text{executed in frame}} C_i \le \Delta t

    This model is especially suitable for:

    * Offline schedulability analysis
    * Deterministic unit testing
    * Worst-case execution modeling
    * TTC (Time-Triggered Cooperative) flight software architectures

    Tasks are instances of :class:`~ADCS.flight_software.tasks.task.Task` and
    communicate via a shared memory dictionary.

    """

    def __init__(self, base_rate_hz: float, tasks: Sequence[Task], memory: dict, debug: bool = False):
        r"""
        Initialize the time-triggered single-core scheduler.

        This constructor configures the base frame rate, task set, and shared
        memory used for inter-task communication. Tasks are sorted internally
        by priority and executed deterministically.

        The base frame duration is computed as:

        .. math::

            \Delta t = \frac{1}{f_{\text{base}}}

        :param base_rate_hz:
            Base scheduler frequency in Hertz.
        :type base_rate_hz: float

        :param tasks:
            List of periodic tasks managed by the scheduler.
            Tasks are sorted by ``priority`` on initialization.
        :type tasks: list[:class:`~ADCS.flight_software.tasks.task.Task`]

        :param memory:
            Shared mutable dictionary acting as an inter-task data bus.
        :type memory: dict

        :param debug:
            If ``True``, enables verbose runtime logging.
        :type debug: bool

        """
        self.base_rate_hz = base_rate_hz
        self.base_dt = 1.0 / base_rate_hz

        self.tasks = sorted(tasks, key=lambda t: t.priority)
        self.memory = memory
        self.debug = debug

        #: Logical flight software time [s]
        self.t_fsw = 0.0

        # Instrumentation
        self._trace: List[FrameEvent] = []
        self._frame_slack: List[Tuple[float, float]] = []  # (t_fsw, slack)


    def step(self):
        r"""
        Advance the scheduler by exactly one base frame.

        This method performs one **deterministic scheduling cycle**:

        #. Reset CPU budget to the base frame duration
        #. Iterate through enabled tasks in priority order
        #. Check task release conditions
        #. Execute tasks while sufficient CPU budget remains
        #. Record WCET-based execution events
        #. Advance logical time by one base frame

        If a task is released but insufficient CPU budget remains, the task is
        skipped for this frame **without consuming its release**. This behavior
        mirrors common TTC implementations.

        Logical time is advanced as:

        .. math::

            t_{\text{fsw}} \leftarrow t_{\text{fsw}} + \Delta t

        :return:
            None
        :rtype:
            None

        """
        cpu_budget = self.base_dt
        frame_start = self.t_fsw

        if self.debug:
            print(f"\n[t = {self.t_fsw:.3f}s] FRAME START (budget={cpu_budget:.4f}s)")

        for task in self.tasks:
            if not task.enabled:
                continue

            released = not (self.t_fsw + 1e-12 < task.next_release_time)
            if not released:
                continue

            if cpu_budget < task.wcet:
                # Not enough CPU budget: skip task this frame (but release is NOT consumed)
                # This is consistent with many TTC implementations that only "consume" a release on execution.
                if self.debug:
                    print(f"  MISS_NO_BUDGET: {task.name} (need={task.wcet:.4f}s have={cpu_budget:.4f}s)")

                self._trace.append(
                    FrameEvent(
                        t_fsw=frame_start,
                        task=task.name,
                        released=True,
                        ran=False,
                        charged_time=0.0,
                        wcet=task.wcet,
                        cpu_budget_before=cpu_budget,
                        cpu_budget_after=cpu_budget,
                        note="MISS_NO_BUDGET",
                    )
                )
                continue

            # Run task
            budget_before = cpu_budget
            if self.debug:
                print(f"  RUN: {task.name} (wcet={task.wcet:.4f}s)")

            task.callback(self.t_fsw, self.memory)
            task.exec_count += 1
            task.next_release_time += task.period

            cpu_budget -= task.wcet

            self._trace.append(
                FrameEvent(
                    t_fsw=frame_start,
                    task=task.name,
                    released=True,
                    ran=True,
                    charged_time=task.wcet,
                    wcet=task.wcet,
                    cpu_budget_before=budget_before,
                    cpu_budget_after=cpu_budget,
                    note="OK",
                )
            )

        self._frame_slack.append((frame_start, cpu_budget))
        self.t_fsw += self.base_dt


    @staticmethod
    def _lcm_int(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    def base_utilization_wcet(self) -> float:
        r"""
        Compute total system utilization using WCET-based analysis.

        Utilization is defined as the sum of each enabled task's WCET divided by
        its period:

        .. math::

            U = \sum_{i} \frac{C_i}{T_i}

        where:

        * :math:`C_i` is the WCET of task *i*
        * :math:`T_i` is the period of task *i*

        This metric provides a **necessary but not sufficient** condition for
        schedulability in TTC systems.

        :return:
            Total WCET-based utilization.
        :rtype:
            float

        """
        return sum((t.wcet / t.period) for t in self.tasks if t.enabled)


    def frame_budget_feasible_wcet(self) -> bool:
        r"""
        Perform a conservative per-frame feasibility check.

        This method assumes **all enabled tasks are released simultaneously** and
        checks whether their combined WCET fits within a single base frame:

        .. math::

            \sum_i C_i \le \Delta t

        While pessimistic, this condition is useful as a fast sanity check during
        system design.

        :return:
            ``True`` if the sum of WCETs fits within one base frame.
        :rtype:
            bool

        """
        sum_wcet = sum(t.wcet for t in self.tasks if t.enabled)
        return sum_wcet <= self.base_dt + 1e-12


    def hyperperiod_s(self, max_den: int = 10_000) -> float:
        r"""
        Compute the approximate hyperperiod of all enabled tasks.

        The hyperperiod is the least common multiple (LCM) of all task periods.
        For real-valued periods, a rational approximation is used:

        .. math::

            H = \operatorname{lcm}(T_1, T_2, \dots, T_n)

        This value represents the time after which the task release pattern
        repeats exactly.

        :param max_den:
            Maximum denominator used when approximating periods as rational numbers.
        :type max_den: int

        :return:
            Hyperperiod in seconds.
        :rtype:
            float

        """
        from fractions import Fraction

        periods = [Fraction(t.period).limit_denominator(max_den) for t in self.tasks if t.enabled]
        if not periods:
            return 0.0

        den_lcm = 1
        for p in periods:
            den_lcm = self._lcm_int(den_lcm, p.denominator)

        ks = [int(p * den_lcm) for p in periods]
        k_lcm = 1
        for k in ks:
            k_lcm = self._lcm_int(k_lcm, k)

        return float(Fraction(k_lcm, den_lcm))

    def reset_trace(self) -> None:
        r"""
        Clear all recorded instrumentation data.

        This method removes:

        * All per-frame :class:`~ADCS.flight_software.schedulers.ttc_single_core.FrameEvent` records
        * All stored per-frame slack measurements

        It does **not** reset task state, logical time, or scheduler configuration.
        This is typically called prior to a new simulation run.

        :return:
            None
        :rtype:
            None

        """
        self._trace.clear()
        self._frame_slack.clear()


    def simulate(self, duration_s: float, reset_trace: bool = True) -> None:
        r"""
        Simulate scheduler execution for a given logical duration.

        The scheduler is stepped repeatedly until the specified duration is
        reached or exceeded. Instrumentation data is collected for analysis.

        The number of frames executed is:

        .. math::

            N = \left\lceil \frac{\text{duration}_s}{\Delta t} \right\rceil

        :param duration_s:
            Logical simulation duration in seconds.
        :type duration_s: float

        :param reset_trace:
            If ``True``, clears previously recorded trace data before simulation.
        :type reset_trace: bool

        :return:
            None
        :rtype:
            None

        """
        if reset_trace:
            self.reset_trace()

        n_steps = int(math.ceil(duration_s / self.base_dt))
        for _ in range(n_steps):
            self.step()

    def stats(self) -> Dict[str, Dict[str, float]]:
        r"""
        Compute per-task execution statistics from the recorded trace.

        Statistics are derived from all recorded
        :class:`~ADCS.flight_software.schedulers.ttc_single_core.FrameEvent` entries
        and include both successful executions and missed executions due to
        insufficient CPU budget.

        For each task, the following metrics are computed:

        +-----------+--------------------------------------------------+
        | Key       | Description                                      |
        +===========+==================================================+
        | attempts  | Number of frames in which the task was released  |
        +-----------+--------------------------------------------------+
        | runs      | Number of frames in which the task executed      |
        +-----------+--------------------------------------------------+
        | misses    | Number of released frames where it did not run   |
        +-----------+--------------------------------------------------+
        | run_rate  | Fraction of attempts that resulted in execution  |
        +-----------+--------------------------------------------------+

        :return:
            Dictionary keyed by task name containing execution statistics.
        :rtype:
            dict[str, dict[str, float]]

        """
        by_task: Dict[str, List[FrameEvent]] = {}
        for e in self._trace:
            by_task.setdefault(e.task, []).append(e)

        out: Dict[str, Dict[str, float]] = {}
        for name, events in by_task.items():
            attempts = len(events)
            runs = sum(1 for e in events if e.ran)
            misses = sum(1 for e in events if (e.released and not e.ran))
            out[name] = {
                "attempts": float(attempts),
                "runs": float(runs),
                "misses": float(misses),
                "run_rate": float(runs) / max(1.0, float(attempts)),
            }
        return out


    def slack_stats(self) -> Dict[str, float]:
        r"""
        Compute slack statistics over all recorded base frames.

        Slack is defined as the unused CPU budget at the end of a base frame:

        .. math::

            S_k = \Delta t - \sum_{i \in \text{executed in frame } k} C_i

        This method reports aggregate statistics over all simulated frames.

        :return:
            Dictionary containing mean and minimum slack values in seconds.
        :rtype:
            dict[str, float]

        """
        if not self._frame_slack:
            return {"mean_slack_s": 0.0, "min_slack_s": 0.0}
        slacks = [s for _, s in self._frame_slack]
        return {
            "mean_slack_s": sum(slacks) / len(slacks),
            "min_slack_s": min(slacks),
        }

    def _frame_slack_lookup(self, frame_t: float) -> float:
        r"""
        Retrieve the recorded slack value for a specific frame start time.

        This helper is primarily used by visualization utilities to annotate
        per-frame diagrams with remaining CPU budget information.

        :param frame_t:
            Logical flight software time corresponding to the frame start.
        :type frame_t: float

        :return:
            Remaining CPU budget (slack) for the specified frame, in seconds.
            Returns ``0.0`` if the frame is not found.
        :rtype:
            float

        """
        for t, s in self._frame_slack:
            if abs(t - frame_t) < 1e-12:
                return s
        return 0.0


    def gantt_ascii(self, t0: float, t1: float, width: int = 80) -> str:
        r"""
        Generate an ASCII Gantt-style timeline of task execution.

        Each line in the output corresponds to one base frame and shows how
        the frame's CPU budget was consumed by tasks, in execution order.
        Tasks are represented by the first letter of their name.

        The horizontal axis is normalized to the base frame duration
        :math:`\Delta t`.

        :param t0:
            Start of the logical time window to display, in seconds.
        :type t0: float

        :param t1:
            End of the logical time window to display, in seconds.
        :type t1: float

        :param width:
            Number of characters used to represent one base frame.
        :type width: int

        :return:
            Multi-line ASCII Gantt diagram.
        :rtype:
            str

        """
        frames: Dict[float, List[FrameEvent]] = {}
        for e in self._trace:
            if t0 <= e.t_fsw <= t1 and e.ran:
                frames.setdefault(e.t_fsw, []).append(e)

        lines = []
        for frame_t in sorted(frames.keys()):
            line = [" "] * width
            cursor = 0.0
            for e in sorted(frames[frame_t], key=lambda x: x.task):
                dur_frac = e.charged_time / self.base_dt if self.base_dt > 0 else 0.0
                a = int(max(0, min(width - 1, (cursor / self.base_dt) * width)))
                b = int(max(a + 1, min(width, ((cursor + e.charged_time) / self.base_dt) * width)))
                ch = e.task[0].upper()
                for i in range(a, b):
                    line[i] = ch
                cursor += e.charged_time

            lines.append(f"{frame_t:8.3f} |{''.join(line)}| slack={self._frame_slack_lookup(frame_t):.4f}s")

        return "\n".join(lines)

    def timeline_dot(self, t0: float, t1: float) -> str:
        r"""
        Generate a Graphviz DOT representation of the execution timeline.

        Each base frame is represented as a node containing:

        * Frame start time
        * Executed tasks and their WCETs
        * Remaining slack for the frame

        Frames are connected sequentially to form a left-to-right timeline.
        The resulting DOT source can be rendered using Graphviz tools.

        :param t0:
            Start of the logical time window to include, in seconds.
        :type t0: float

        :param t1:
            End of the logical time window to include, in seconds.
        :type t1: float

        :return:
            Graphviz DOT source encoding the execution timeline.
        :rtype:
            str

        """
        frames: Dict[float, List[FrameEvent]] = {}
        for e in self._trace:
            if t0 <= e.t_fsw <= t1 and e.ran:
                frames.setdefault(e.t_fsw, []).append(e)

        out = [
            "digraph schedule {",
            "  rankdir=LR;",
            "  node [shape=record, fontname=Helvetica];",
        ]

        prev = None
        for i, ft in enumerate(sorted(frames.keys())):
            events = frames[ft]
            fields = [f"{e.task}\\n{e.charged_time:.4f}s" for e in events]
            slack = self._frame_slack_lookup(ft)
            label = " | ".join(fields) + f" | slack\\n{slack:.4f}s"
            node_name = f"frame_{i}"
            out.append(f'  {node_name} [label="{{t={ft:.3f}s | {label}}}"];')
            if prev is not None:
                out.append(f"  {prev} -> {node_name};")
            prev = node_name

        out.append("}")
        return "\n".join(out)

    # ---------------------------
    # Reporting helpers
    # ---------------------------
    def report(self) -> str:
        r"""
        Generate a human-readable schedulability and execution report.

        The report summarizes:

        * Base frame parameters
        * WCET-based utilization
        * Approximate hyperperiod
        * Mean and minimum slack
        * Per-task execution and miss statistics

        This method relies on trace data generated by
        :meth:`~ADCS.flight_software.schedulers.ttc_single_core.TTC_Single_Core.simulate`.

        :return:
            Multi-line formatted report string.
        :rtype:
            str

        """
        U = self.base_utilization_wcet()
        slack = self.slack_stats()
        hp = self.hyperperiod_s()

        lines = []
        lines.append("=== TTC Single-Core Schedulability Report (WCET-accounted) ===")
        lines.append(f"Base rate: {self.base_rate_hz:.3f} Hz  (base_dt={self.base_dt:.6f} s)")
        lines.append(f"Enabled tasks: {sum(1 for t in self.tasks if t.enabled)} / {len(self.tasks)}")
        lines.append(f"Utilization (WCET): U = sum(C/T) = {U:.4f}")
        lines.append(f"Hyperperiod (approx): {hp:.6f} s")
        lines.append(f"Mean slack: {slack['mean_slack_s']:.6f} s, Min slack: {slack['min_slack_s']:.6f} s")
        lines.append(f"Frame-feasible (sum WCET <= base_dt): {self.frame_budget_feasible_wcet()}")

        st = self.stats()
        if st:
            lines.append("\nPer-task trace stats (from last simulation):")
            for name in sorted(st.keys()):
                d = st[name]
                lines.append(
                    f"  - {name}: attempts={int(d['attempts'])} "
                    f"runs={int(d['runs'])} "
                    f"misses={int(d['misses'])} "
                    f"run_rate={d['run_rate']*100:.1f}%"
                )
        else:
            lines.append("\n(No trace data yet. Run simulate(...) first.)")

        return "\n".join(lines)
