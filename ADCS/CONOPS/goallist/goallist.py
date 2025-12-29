__all__ = ["GoalList"]

import numpy as np
from typing import Dict, List, Tuple

import bisect
from ..goals import Goal, No_Goal
from ADCS.orbits.orbital_state import Orbital_State

class GoalList:
    """
    Timeline-based container for attitude or mission goals.

    The :class:`GoalList` class manages a time-ordered sequence of
    :class:`~ADCS.goals.goal.Goal` objects. Each goal becomes *active*
    at a specified mission time and remains active until superseded
    by a later goal.

    Internally, the goals are stored as two synchronized sorted lists:
    one for activation times and one for the corresponding goals.
    Efficient lookup of the active goal is performed using binary
    search (:mod:`bisect`).

    This class is typically queried at every guidance or control
    update step to determine the currently active reference definition.

    Parameters
    ----------
    goal_timeline : Dict[float, Goal], optional
        Dictionary mapping activation times (in seconds) to
        :class:`~ADCS.goals.goal.Goal` objects. The dictionary is
        automatically sorted by time during initialization.

    Attributes
    ----------
    times : List[float]
        Sorted list of activation times.
    goals : List[Goal]
        List of goals corresponding to each activation time.

    Notes
    -----
    If no goals are defined, calls to
    :meth:`get_active_goal` return a
    :class:`~ADCS.goals.no_goal.No_Goal` instance.

    All times are assumed to be expressed in a consistent mission
    time reference (e.g. seconds since epoch).

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.no_goal.No_Goal`
    """
    def __init__(self, goal_timeline: Dict[float, Goal] = None) -> None:
        """
        Initialize a goal timeline.

        Parameters
        ----------
        goal_timeline : Dict[float, Goal], optional
            Mapping from activation times to
            :class:`~ADCS.goals.goal.Goal` objects. If provided, the
            timeline is sorted in ascending order of time.

        Notes
        -----
        If ``goal_timeline`` is ``None`` or empty, the instance starts
        with no defined goals.
        """
        self.times: List[float] = []
        self.goals: List[Goal] = []

        if goal_timeline:
            sorted_items = sorted(goal_timeline.items())
            self.times = [t for t, g in sorted_items]
            self.goals = [g for t, g in sorted_items]

    def add_goal(self, time: float, goal: Goal) -> None:
        """
        Insert or update a goal at a given activation time.

        This method inserts a new goal into the timeline while maintaining
        time ordering. If a goal already exists at the specified time
        (within numerical tolerance), it is replaced.

        Parameters
        ----------
        time : float
            Activation time of the goal.
        goal : Goal
            Goal to activate at ``time``.

        Notes
        -----
        Time equality is checked using a tolerance of :math:`10^{-9}` to
        avoid floating-point comparison issues.

        Complexity
        ----------
        :math:`\mathcal{O}(N)` due to list insertion.
        """
        idx = bisect.bisect_left(self.times, time)

        if idx < len(self.times) and abs(self.times[idx] - time) < 1e-9:
            self.goals[idx] = goal
        else:
            self.times.insert(idx, time)
            self.goals.insert(idx, goal)

    def get_active_goal(self, t: float) -> Goal:
        """
        Return the goal active at a given time.

        The active goal is defined as the most recent goal whose activation
        time is less than or equal to ``t``.

        Parameters
        ----------
        t : float
            Query time.

        Returns
        -------
        Goal
            The active :class:`~ADCS.goals.goal.Goal` at time ``t``.
            If no goals are defined, a
            :class:`~ADCS.goals.no_goal.No_Goal` instance is returned.

        Notes
        -----
        If ``t`` is earlier than the first activation time, the first
        goal in the timeline is returned.

        This method uses binary search and runs in
        :math:`\mathcal{O}(\log N)` time.
        """
        if not self.times:
            return No_Goal()
        
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return self.goals[0]
        return self.goals[idx]
    
    def to_ref(self, t: float, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reference definition associated with the active goal.

        This method delegates reference generation to the currently active
        :class:`~ADCS.goals.goal.Goal`.

        Parameters
        ----------
        t : float
            Current mission time.
        os0 : Orbital_State
            Current orbital state, provided as an
            :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Reference state returned by the active goal. The exact meaning
            of the arrays depends on the concrete goal implementation
            (e.g. attitude reference and angular rate reference).

        See Also
        --------
        :meth:`ADCS.goals.goal.Goal.to_ref`
        """
        active_goal = self.get_active_goal(t)
        return active_goal.to_ref(os0)
