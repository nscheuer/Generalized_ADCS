__all__ = ["GoalList"]

import numpy as np
from typing import Dict, List, Tuple

import bisect
from ..goals import Goal, No_Goal
from ADCS.orbits.orbital_state import Orbital_State

class GoalList:
    def __init__(self, goal_timeline: Dict[float, Goal] = None) -> None:
        self.times: List[float] = []
        self.goals: List[Goal] = []

        if goal_timeline:
            sorted_items = sorted(goal_timeline.items())
            self.times = [t for t, g in sorted_items]
            self.goals = [g for t, g in sorted_items]

    def add_goal(self, time: float, goal: Goal) -> None:
        idx = bisect.bisect_left(self.times, time)

        if idx < len(self.times) and abs(self.times[idx] - time) < 1e-9:
            self.goals[idx] = goal
        else:
            self.times.insert(idx, time)
            self.goals.insert(idx, goal)

    def get_active_goal(self, t: float) -> Goal:
        if not self.times:
            return No_Goal()
        
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return self.goals[0]
        return self.goals[idx]
    
    def to_ref(self, t: float, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        active_goal = self.get_active_goal(t)
        return active_goal.to_ref(os0)
