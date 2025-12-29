__all__ = ["TTC_Single_Core"]

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ADCS.flight_software.tasks.task import Task


@dataclass
class FrameEvent:
    r"""
    Record of one task execution within a scheduler frame (WCET-accounted).

    Attributes
    ----------
    t_fsw : float
        Frame start logical time [s].
    task : str
        Task name.
    released : bool
        Whether the task was eligible for execution (release time met).
    ran : bool
        Whether the callback was executed.
    charged_time : float
        CPU time charged to the task in this model [s] (equal to WCET).
    wcet : float
        Task WCET [s].
    cpu_budget_before : float
        Remaining budget before executing the task [s].
    cpu_budget_after : float
        Remaining budget after executing the task [s].
    note : str
        Human-readable note ("OK", "MISS_NO_BUDGET").
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
    Single-core time-triggered scheduler (WCET-accounted, deterministic).

    This class implements a **cooperative, non-preemptive, fixed-priority**
    scheduler intended to model a single-core spacecraft flight computer.
    Tasks are released periodically and executed within a fixed base frame.

    This variant uses **WCET-based accounting**:
    each executed task consumes exactly its configured WCET from the frame budget.
    This makes simulations deterministic and suitable for unit tests and
    schedulability analysis.

    The scheduler advances in discrete frames of duration:

    .. math::

       \Delta t = \frac{1}{f_{\text{base}}}

    At each frame, tasks are executed in ascending priority order
    (lower value means higher priority), subject to CPU budget.

    The CPU usage within a frame satisfies:

    .. math::

       \sum_i C_i \le \Delta t

    where \( C_i \) are the WCETs of tasks executed in that frame.
    """

    def __init__(self, base_rate_hz: float, tasks: List[Task], memory: dict, debug: bool = False):
        r"""
        Initialize the WCET-accounted scheduler.

        Parameters
        ----------
        base_rate_hz : float
            Base scheduler frequency in Hertz.

        tasks : List[Task]
            List of tasks managed by the scheduler (sorted by priority).

        memory : dict
            Shared memory dictionary used as an inter-task data bus.

        debug : bool, optional
            Enable verbose debug logging.
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

    # ---------------------------
    # Runtime scheduler step
    # ---------------------------
    def step(self):
        r"""
        Advance the scheduler by one base frame.

        Budgeting is WCET-based and deterministic: when a task runs, it consumes
        exactly ``task.wcet`` from the remaining frame budget.
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

    # ---------------------------
    # Schedulability analysis tools
    # ---------------------------
    @staticmethod
    def _lcm_int(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    def base_utilization_wcet(self) -> float:
        r"""
        Compute total utilization using WCET.

        .. math::

            U = \sum_i \frac{C_i}{T_i}

        Returns
        -------
        float
            Total utilization.
        """
        return sum((t.wcet / t.period) for t in self.tasks if t.enabled)

    def frame_budget_feasible_wcet(self) -> bool:
        r"""
        Conservative check: if all tasks released simultaneously, would their WCET sum fit in one frame?

        Returns
        -------
        bool
            True if \(\sum C_i \le \Delta t\) under the "all released" assumption.
        """
        sum_wcet = sum(t.wcet for t in self.tasks if t.enabled)
        return sum_wcet <= self.base_dt + 1e-12

    def hyperperiod_s(self, max_den: int = 10_000) -> float:
        r"""
        Compute the hyperperiod (least common multiple) of task periods.

        Parameters
        ----------
        max_den : int, optional
            Maximum denominator for rational approximation.

        Returns
        -------
        float
            Hyperperiod in seconds.
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
        r"""Clear collected instrumentation trace and slack history."""
        self._trace.clear()
        self._frame_slack.clear()

    def simulate(self, duration_s: float, reset_trace: bool = True) -> None:
        r"""
        Run the scheduler for a specified logical duration.

        Parameters
        ----------
        duration_s : float
            Logical duration to simulate [s].
        reset_trace : bool, optional
            If True, clears previous trace before running.
        """
        if reset_trace:
            self.reset_trace()

        n_steps = int(math.ceil(duration_s / self.base_dt))
        for _ in range(n_steps):
            self.step()

    def stats(self) -> Dict[str, Dict[str, float]]:
        r"""
        Compute per-task statistics from the recorded trace.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Per-task counts for runs and misses.
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
        Compute slack statistics per base frame.

        Returns
        -------
        Dict[str, float]
            Contains mean/min slack over recorded frames.
        """
        if not self._frame_slack:
            return {"mean_slack_s": 0.0, "min_slack_s": 0.0}
        slacks = [s for _, s in self._frame_slack]
        return {
            "mean_slack_s": sum(slacks) / len(slacks),
            "min_slack_s": min(slacks),
        }

    # ---------------------------
    # Diagram / visualization helpers
    # ---------------------------
    def _frame_slack_lookup(self, frame_t: float) -> float:
        for t, s in self._frame_slack:
            if abs(t - frame_t) < 1e-12:
                return s
        return 0.0

    def gantt_ascii(self, t0: float, t1: float, width: int = 80) -> str:
        r"""
        Generate an ASCII Gantt-like view for frames between t0 and t1.

        Parameters
        ----------
        t0, t1 : float
            Logical time window [s].
        width : int, optional
            Characters across one frame line.

        Returns
        -------
        str
            Multi-line ASCII diagram.
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
        Generate a Graphviz DOT timeline diagram for a time window.

        Parameters
        ----------
        t0, t1 : float
            Logical time window [s].

        Returns
        -------
        str
            Graphviz DOT source encoding a frame-by-frame execution timeline.
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
        Create a human-readable schedulability report.

        Returns
        -------
        str
            Multi-line report with utilization and trace-based stats.
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
