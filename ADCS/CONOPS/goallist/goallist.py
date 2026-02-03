from __future__ import annotations

__all__ = ["GoalList"]

import bisect
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

from ..goals import Goal, No_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants

_TimeUnits = Literal["centuries", "seconds"]


class GoalList:
    r"""
    Timeline-based container for attitude or mission goals.

    This class implements a piecewise-constant goal selection function
    over mission time, with *internal storage in Julian centuries*.

    Two primary usage modes are supported:

    1) **Julian-centuries mode** (default)
       Activation/query times are provided directly in Julian centuries,
       and stored as-is.

    2) **Seconds mode**
       Activation/query times are provided in seconds relative to a
       specified start epoch in Julian centuries, ``start_juliantime``.
       Times are converted internally using

       .. math::

           T = T_0 + t_{\text{sec}} \cdot c_{\text{sec}\to\text{cent}}

       where :math:`T` is Julian centuries, :math:`T_0` is
       ``start_juliantime``, and :math:`c_{\text{sec}\to\text{cent}}`
       is :attr:`~ADCS.orbits.universal_constants.TimeConstants.sec2cent`.

    Let a finite ordered set of activation times (in Julian centuries) be

    .. math::

        \mathcal{T} = \{ T_0, T_1, \dots, T_{N-1} \}, \quad
        T_0 < T_1 < \dots < T_{N-1}

    with an associated set of goals

    .. math::

        \mathcal{G} = \{ G_0, G_1, \dots, G_{N-1} \}

    where each :math:`G_i` is an instance of
    :class:`~ADCS.goals.goal.Goal`.

    The active goal as a function of time :math:`T` is defined as

    .. math::

        G(T) = G_k \quad \text{with} \quad
        k = \max \{ i \mid T_i \le T \}

    If no such index exists, the first goal :math:`G_0` is returned.
    If the timeline is empty, a
    :class:`~ADCS.goals.no_goal.No_Goal` instance is returned.

    Internally, the timeline is represented by two synchronized
    sorted lists: one containing activation times (Julian centuries)
    and one containing the corresponding goals. Binary search is used
    to ensure efficient lookup.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.no_goal.No_Goal`

    """

    def __init__(
        self,
        goal_timeline: Optional[Dict[float, Goal]] = None,
        *,
        time_units: _TimeUnits = "centuries",
        start_juliantime: Optional[float] = None,
    ) -> None:
        r"""
        Initialize a goal timeline.

        The input mapping is sorted in ascending order of activation
        time and stored internally as ordered lists *in Julian centuries*.

        Two initialization standards are supported:

        **A) Julian-centuries mode** (default)
            - ``time_units="centuries"``
            - ``goal_timeline`` keys are interpreted as Julian centuries.
            - ``start_juliantime`` must be ``None``.

        **B) Seconds mode**
            - ``time_units="seconds"``
            - ``goal_timeline`` keys are interpreted as seconds.
            - ``start_juliantime`` must be provided (Julian centuries).
            - Internal activation times are computed via

              .. math::

                  T_i = T_{\text{start}} + t_i \cdot c_{\text{sec}\to\text{cent}}

        Let the (Julian-centuries) internal dictionary be

        .. math::

            \{ (T_i, G_i) \}_{i=0}^{N-1}

        After initialization, the internal state satisfies

        .. math::

            \texttt{times}[i] = T_i, \quad
            \texttt{goals}[i] = G_i

        with :math:`T_i < T_{i+1}`.

        :param goal_timeline:
            Mapping from activation times to
            :class:`~ADCS.goals.goal.Goal` objects. The meaning of the keys
            depends on ``time_units`` (Julian centuries or seconds).
        :type goal_timeline:
            Dict[float, Goal] or None

        :param time_units:
            Default time unit for inputs to this instance. Use ``"centuries"``
            for direct Julian-centuries inputs, or ``"seconds"`` for seconds
            relative to ``start_juliantime``.
        :type time_units:
            Literal["centuries","seconds"]

        :param start_juliantime:
            Start epoch in Julian centuries, required when ``time_units="seconds"``.
            Also required when calling methods with ``time_units="seconds"``.
        :type start_juliantime:
            float or None

        :return:
            None
        :rtype:
            None

        """
        self.time_units: _TimeUnits = str(time_units).strip().lower()  # type: ignore[assignment]
        if self.time_units not in ("centuries", "seconds"):
            raise ValueError("time_units must be one of: 'centuries', 'seconds'")

        if start_juliantime is not None:
            self.start_juliantime: Optional[float] = float(start_juliantime)
        else:
            self.start_juliantime = None

        # Enforce constructor compatibility:
        # - centuries default mode: start_juliantime may be None or provided (allowed),
        #   since it can be useful later for seconds-based method calls.
        # - seconds default mode: start_juliantime must be provided.
        if self.time_units == "seconds" and self.start_juliantime is None:
            raise ValueError("start_juliantime is required when time_units='seconds'.")

        # Internal storage is ALWAYS in Julian centuries.
        self.times: List[float] = []
        self.goals: List[Goal] = []

        if goal_timeline:
            items = list(goal_timeline.items())
            for k, g in items:
                try:
                    _ = float(k)
                except Exception as e:
                    raise TypeError(
                        f"Goal timeline time key {k!r} is not a float-like value."
                    ) from e
                if g is None:
                    raise ValueError(f"Goal timeline contains None goal at time {k!r}.")

            sorted_items = sorted(items, key=lambda kv: float(kv[0]))

            # Interpret keys according to the constructor's default time_units.
            if self.time_units == "centuries":
                self.times = [float(t) for t, _g in sorted_items]
                self.goals = [g for _t, g in sorted_items]
            else:
                self._require_start_juliantime_for_seconds()
                self.times = [self._sec_to_cent(float(t)) for t, _g in sorted_items]
                self.goals = [g for _t, g in sorted_items]

            self._assert_strictly_increasing(self.times)

    # ---------------- Public API ----------------

    def add_goal(self, time: float, goal: Goal, *, time_units: Optional[_TimeUnits] = None) -> None:
        r"""
        Insert or update a goal at a given activation time.

        The method maintains the strict ordering of activation times in the
        *internal Julian-centuries timeline*.

        By default, the instance's ``time_units`` is used to interpret ``time``.
        Alternatively, ``time_units`` may be provided per-call to override the
        interpretation.

        - If ``time_units="centuries"``: ``time`` is interpreted as a Julian-centuries value.
        - If ``time_units="seconds"``: ``time`` is interpreted as seconds relative to
          ``start_juliantime`` and is converted internally.

        Given an insertion time :math:`T` (in Julian centuries), the new goal is
        placed such that the ordered set

        .. math::

            T_0 < T_1 < \dots < T < \dots < T_{N}

        is preserved.

        If an existing activation time :math:`T_i` satisfies

        .. math::

            |T_i - T| < 10^{-12}

        then the goal :math:`G_i` is replaced instead of inserting a new entry.

        :param time:
            Activation time of the goal, interpreted according to ``time_units``.
        :type time:
            float

        :param goal:
            Goal instance to activate at the specified time.
        :type goal:
            Goal

        :param time_units:
            Optional per-call override of time units. Must be ``"centuries"`` or ``"seconds"``.
            If omitted, the instance default ``self.time_units`` is used.
        :type time_units:
            Literal["centuries","seconds"] or None

        :return:
            None
        :rtype:
            None

        """
        if goal is None:
            raise ValueError("goal must not be None")

        T = self._to_centuries(float(time), time_units=time_units)

        idx = bisect.bisect_left(self.times, T)

        if idx < len(self.times) and abs(self.times[idx] - T) < 1e-12:
            self.goals[idx] = goal
        else:
            self.times.insert(idx, T)
            self.goals.insert(idx, goal)

        self._assert_strictly_increasing(self.times)

    def get_active_goal(self, t: float, *, time_units: Optional[_TimeUnits] = None) -> Goal:
        r"""
        Return the goal active at a given time.

        This method evaluates the piecewise-constant mapping

        .. math::

            G(T) = \arg\max_{G_i} \{ T_i \le T \}

        using binary search on the internal Julian-centuries timeline.
        Computational complexity is

        .. math::

            \mathcal{O}(\log N)

        where :math:`N` is the number of defined goals.

        By default, the instance's ``time_units`` is used to interpret ``t``.
        Alternatively, ``time_units`` may be provided per-call to override the
        interpretation.

        - If ``time_units="centuries"``: ``t`` is a Julian-centuries value.
        - If ``time_units="seconds"``: ``t`` is seconds relative to
          ``start_juliantime`` and is converted internally.

        If the timeline is empty, a
        :class:`~ADCS.goals.no_goal.No_Goal` instance is returned.

        If :math:`T < T_0`, the first goal :math:`G_0` is returned.

        :param t:
            Query time, interpreted according to ``time_units``.
        :type t:
            float

        :param time_units:
            Optional per-call override of time units. Must be ``"centuries"`` or ``"seconds"``.
            If omitted, the instance default ``self.time_units`` is used.
        :type time_units:
            Literal["centuries","seconds"] or None

        :return:
            The active goal at time ``t``.
        :rtype:
            Goal

        """
        if not self.times:
            return No_Goal()

        T = self._to_centuries(float(t), time_units=time_units)

        idx = bisect.bisect_right(self.times, T) - 1
        if idx < 0:
            return self.goals[0]
        return self.goals[idx]

    def to_ref(
        self,
        t: float,
        os0: Orbital_State,
        *,
        time_units: Optional[_TimeUnits] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the reference definition associated with the active goal.

        This method first determines the active goal :math:`G(T)` for the
        internal Julian-centuries time corresponding to the provided ``t``, and
        then delegates reference generation to

        .. math::

            ( \mathbf{r}_{\text{ref}}, \boldsymbol{\omega}_{\text{ref}} )
            = G(T).\texttt{to\_ref}( \mathcal{O}(T) )

        where :math:`\mathcal{O}(\cdot)` denotes the current orbital state.

        The semantic meaning of the returned arrays is defined by the
        concrete implementation of
        :class:`~ADCS.goals.goal.Goal`.

        .. note::

            ``GoalList`` does not modify ``os0``. The time argument ``t`` is
            used only to select which goal is active.

        By default, the instance's ``time_units`` is used to interpret ``t``.
        Alternatively, ``time_units`` may be provided per-call to override the
        interpretation.

        :param t:
            Current mission time, interpreted according to ``time_units``.
        :type t:
            float

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :param time_units:
            Optional per-call override of time units. Must be ``"centuries"`` or ``"seconds"``.
            If omitted, the instance default ``self.time_units`` is used.
        :type time_units:
            Literal["centuries","seconds"] or None

        :return:
            Reference definition produced by the active goal.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        active_goal = self.get_active_goal(t, time_units=time_units)
        return active_goal.to_ref(os0)

    # ---------------- Internals ----------------

    def _to_centuries(self, t: float, *, time_units: Optional[_TimeUnits]) -> float:
        """
        Convert a time value to Julian centuries (internal representation),
        using either the per-call override or the instance default.
        """
        units = self.time_units if time_units is None else str(time_units).strip().lower()
        if units not in ("centuries", "seconds"):
            raise ValueError("time_units must be one of: 'centuries', 'seconds'")

        if units == "centuries":
            return float(t)

        # seconds mode
        self._require_start_juliantime_for_seconds()
        return self._sec_to_cent(float(t))

    def _sec_to_cent(self, t_sec: float) -> float:
        """
        Convert seconds relative to start_juliantime into absolute Julian centuries.
        """
        # start_juliantime is validated by _require_start_juliantime_for_seconds
        return float(self.start_juliantime) + float(t_sec) * float(TimeConstants.sec2cent)  # type: ignore[arg-type]

    def _require_start_juliantime_for_seconds(self) -> None:
        """
        Ensure start_juliantime is available for seconds->centuries conversion.
        """
        if self.start_juliantime is None:
            raise ValueError(
                "start_juliantime must be provided to use time_units='seconds' "
                "(either in the constructor or before calling with time_units='seconds')."
            )

    @staticmethod
    def _assert_strictly_increasing(times: List[float]) -> None:
        """
        Ensure internal activation times are strictly increasing.
        """
        if len(times) <= 1:
            return
        for i in range(1, len(times)):
            if not (times[i] > times[i - 1]):
                raise ValueError(
                    "Goal activation times must be strictly increasing after conversion "
                    f"(found times[{i-1}]={times[i-1]!r}, times[{i}]={times[i]!r})."
                )
