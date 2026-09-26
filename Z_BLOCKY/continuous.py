"""Continuous plant dynamics and two ways to retain sampled state history."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from .blocky import Block, State


@dataclass(frozen=True, slots=True)
class Dynamics:
    """Derivative of one plant state; controls are held between publications."""

    state_name: str
    derivative: Callable[[float, np.ndarray, Mapping[str, Any]], np.ndarray]
    max_step_s: float = 0.1

    def __post_init__(self) -> None:
        if self.max_step_s <= 0:
            raise ValueError("max_step_s must be positive")


class StateSampledBlock(Block):
    """Sensor sampling a continuous state at its nominal capture time.

    The sample must be pure: the scheduler may evaluate it later, once it
    knows which completed capture a consumer will actually read.
    """

    __slots__ = ("state_name",)

    def __init__(self, name: str, *, state_name: str, **kwargs: Any) -> None:
        super().__init__(name, sample_on_demand=True, **kwargs)
        self.state_name = state_name

    def step(self, inputs, context=None):
        return self.sample(context.state_at(self.state_name), inputs, context)

    def sample(self, state: np.ndarray, inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        raise NotImplementedError


class ContinuousRuntime:
    """Advances one state vector and retains either dense intervals or snapshots."""

    def __init__(self, state: State, dynamics: Dynamics, *, history: str) -> None:
        if history not in ("dense", "cache"):
            raise ValueError("history must be 'dense' or 'cache'")
        initial = np.asarray(state.initial, dtype=float)
        if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
            raise ValueError("continuous state initial value must be a finite, nonempty 1-D vector")
        self.state_name = state.name
        self.dynamics = dynamics
        self.history = history
        self.time_s = 0.0
        self.value = initial.copy()
        self.controls: dict[str, Any] = {}
        self._ends: list[float] = []
        self._intervals: list[tuple[float, float, Any]] = []
        self._samples: dict[float, np.ndarray] = {}
        self.max_history_items = 0

    def advance_to(self, at: float) -> None:
        if at < self.time_s - 1e-12:
            raise ValueError("continuous state cannot run backwards")
        if at <= self.time_s:
            return
        start = self.time_s
        solution = solve_ivp(
            lambda t, y: self.dynamics.derivative(t, y, self.controls),
            (start, at), self.value, rtol=1e-8, atol=1e-10,
            max_step=self.dynamics.max_step_s,
            dense_output=self.history == "dense",
        )
        if not solution.success:
            raise RuntimeError(f"integration failed: {solution.message}")
        self.value = solution.y[:, -1].copy()
        self.time_s = at
        if self.history == "dense":
            self._ends.append(at)
            self._intervals.append((start, at, solution.sol))
            self.max_history_items = max(self.max_history_items, len(self._intervals))

    def capture(self, at: float) -> None:
        if self.history == "cache":
            self._samples[at] = self.value.copy()
            self.max_history_items = max(self.max_history_items, len(self._samples))

    def state_at(self, at: float) -> np.ndarray:
        if abs(at - self.time_s) <= 1e-12:
            return self.value.copy()
        if self.history == "cache":
            return self._samples[at].copy()
        index = bisect_right(self._ends, at)
        if index >= len(self._intervals):
            index = len(self._intervals) - 1
        if index < 0 or at < self._intervals[index][0] - 1e-12:
            raise KeyError(f"state at {at:g} s is no longer in history")
        return np.asarray(self._intervals[index][2](at), dtype=float).copy()

    def prune_before(self, at: float) -> None:
        """Discard captures older than the latest materialized sample."""
        if self.history == "cache":
            for timestamp in tuple(self._samples):
                if timestamp < at - 1e-12:
                    del self._samples[timestamp]
        else:
            while self._intervals and self._intervals[0][1] < at - 1e-12:
                self._intervals.pop(0)
                self._ends.pop(0)

    @property
    def history_items(self) -> int:
        return len(self._intervals) if self.history == "dense" else len(self._samples)
