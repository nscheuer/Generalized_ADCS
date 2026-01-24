__all__ = ["GoalList"]

import numpy as np
from typing import Dict, List, Tuple

import bisect
from ..goals import Goal, No_Goal
from ADCS.orbits.orbital_state import Orbital_State

class GoalList:
    r"""
    Timeline-based container for attitude or mission goals.

    This class implements a piecewise-constant goal selection function
    over mission time. Let a finite ordered set of activation times be

    .. math::

        \mathcal{T} = \{ t_0, t_1, \dots, t_{N-1} \}, \quad t_0 < t_1 < \dots < t_{N-1}

    with an associated set of goals

    .. math::

        \mathcal{G} = \{ G_0, G_1, \dots, G_{N-1} \}

    where each :math:`G_i` is an instance of
    :class:`~ADCS.goals.goal.Goal`.

    The active goal as a function of time :math:`t` is defined as

    .. math::

        G(t) = G_k \quad \text{with} \quad
        k = \max \{ i \mid t_i \le t \}

    If no such index exists, the first goal :math:`G_0` is returned.
    If the timeline is empty, a
    :class:`~ADCS.goals.no_goal.No_Goal` instance is returned.

    Internally, the timeline is represented by two synchronized
    sorted lists: one containing activation times and one containing
    the corresponding goals. Binary search is used to ensure efficient
    lookup.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.no_goal.No_Goal`

    """
    def __init__(self, goal_timeline: Dict[float, Goal] = None) -> None:
        r"""
        Initialize a goal timeline.

        The input mapping is sorted in ascending order of activation
        time and stored internally as ordered lists.

        Let the input dictionary be

        .. math::

            \{ (t_i, G_i) \}_{i=0}^{N-1}

        After initialization, the internal state satisfies

        .. math::

            \texttt{times}[i] = t_i, \quad
            \texttt{goals}[i] = G_i

        with :math:`t_i < t_{i+1}`.

        :param goal_timeline:
            Mapping from activation times in seconds to
            :class:`~ADCS.goals.goal.Goal` objects.
        :type goal_timeline:
            Dict[float, Goal] or None

        :return:
            None
        :rtype:
            None

        """
        self.times: List[float] = []
        self.goals: List[Goal] = []

        if goal_timeline:
            sorted_items = sorted(goal_timeline.items())
            self.times = [t for t, g in sorted_items]
            self.goals = [g for t, g in sorted_items]

    def add_goal(self, time: float, goal: Goal) -> None:
        r"""
        Insert or update a goal at a given activation time.

        The method maintains the strict ordering of activation times.
        Given an insertion time :math:`t`, the new goal is placed such
        that the ordered set

        .. math::

            t_0 < t_1 < \dots < t < \dots < t_{N}

        is preserved.

        If an existing activation time :math:`t_i` satisfies

        .. math::

            |t_i - t| < 10^{-9}

        then the goal :math:`G_i` is replaced instead of inserting a
        new entry.

        The active-goal function :math:`G(t)` is therefore updated
        only for times :math:`t' \ge t`.

        :param time:
            Activation time of the goal in seconds.
        :type time:
            float

        :param goal:
            Goal instance to activate at the specified time.
        :type goal:
            Goal

        :return:
            None
        :rtype:
            None

        """
        idx = bisect.bisect_left(self.times, time)

        if idx < len(self.times) and abs(self.times[idx] - time) < 1e-9:
            self.goals[idx] = goal
        else:
            self.times.insert(idx, time)
            self.goals.insert(idx, goal)

    def get_active_goal(self, t: float) -> Goal:
        r"""
        Return the goal active at a given time.

        This method evaluates the piecewise-constant mapping

        .. math::

            G(t) = \arg\max_{G_i} \{ t_i \le t \}

        using binary search. Computational complexity is

        .. math::

            \mathcal{O}(\log N)

        where :math:`N` is the number of defined goals.

        If the timeline is empty, a
        :class:`~ADCS.goals.no_goal.No_Goal` instance is returned.

        If :math:`t < t_0`, the first goal :math:`G_0` is returned.

        :param t:
            Query time in seconds.
        :type t:
            float

        :return:
            The active goal at time ``t``.
        :rtype:
            Goal

        """
        if not self.times:
            return No_Goal()
        
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return self.goals[0]
        return self.goals[idx]
    
    def to_ref(self, t: float, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the reference definition associated with the active goal.

        This method first determines the active goal :math:`G(t)` and
        then delegates reference generation to

        .. math::

            ( \mathbf{r}_{\text{ref}}, \boldsymbol{\omega}_{\text{ref}} )
            = G(t).\texttt{to\_ref}( \mathcal{O}(t) )

        where :math:`\mathcal{O}(t)` denotes the current orbital state.

        The semantic meaning of the returned arrays is defined by the
        concrete implementation of
        :class:`~ADCS.goals.goal.Goal`.

        :param t:
            Current mission time in seconds.
        :type t:
            float

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Reference definition produced by the active goal.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        active_goal = self.get_active_goal(t)
        return active_goal.to_ref(os0)
