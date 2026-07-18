"""Typed spacecraft attitude-state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


def _vector(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


@dataclass(slots=True)
class State:
    """Physical spacecraft state ``[w, q, h]``.

    The class deliberately does not emulate a NumPy array. Numerical-library
    boundaries must use :meth:`as_array` explicitly.
    """

    w: np.ndarray
    q: np.ndarray
    h: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def __post_init__(self) -> None:
        self.w = _vector(self.w, name="w", size=3)
        self.q = _vector(self.q, name="q", size=4)
        self.h = _vector(self.h, name="h")

    @classmethod
    def from_array(cls, value: Any) -> State:
        """Build a state from the established ``[w(3), q(4), h]`` ordering."""
        array = _vector(value, name="state")
        if array.size < 7:
            raise ValueError(f"state must contain at least 7 values, got {array.size}")
        return cls(w=array[:3], q=array[3:7], h=array[7:])

    def as_array(self) -> np.ndarray:
        """Return an owned physical-state vector in ``[w, q, h]`` ordering."""
        return np.concatenate((self.w, self.q, self.h))

    def copy(self) -> State:
        return State(w=self.w, q=self.q, h=self.h)

    def normalized(self) -> State:
        """Return a copy with a unit quaternion, without changing this state."""
        norm = float(np.linalg.norm(self.q))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("q must have a finite, non-zero norm to normalize")
        return State(w=self.w, q=self.q / norm, h=self.h)

    def to_dict(self) -> dict[str, Any]:
        return {"w": self.w.tolist(), "q": self.q.tolist(), "h": self.h.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> State:
        return cls(w=payload["w"], q=payload["q"], h=payload.get("h", ()))

    @staticmethod
    def stack(states: Iterable[State]) -> np.ndarray:
        rows = [state.as_array() for state in states]
        if not rows:
            return np.empty((0, 0), dtype=float)
        widths = {row.size for row in rows}
        if len(widths) != 1:
            raise ValueError("all states must have the same number of reaction-wheel states")
        return np.vstack(rows)


@dataclass(slots=True)
class EstimatedState(State):
    """Physical state plus estimated parameters and uncertainty."""

    act_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    sens_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    dist_param: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    cov: np.ndarray | None = None
    int_cov: np.ndarray | None = None

    def __post_init__(self) -> None:
        State.__post_init__(self)
        self.act_bias = _vector(self.act_bias, name="act_bias")
        self.sens_bias = _vector(self.sens_bias, name="sens_bias")
        self.dist_param = _vector(self.dist_param, name="dist_param")

        reduced_size = self.augmented_size - 1
        if self.cov is None:
            self.cov = np.zeros((reduced_size, reduced_size), dtype=float)
        else:
            self.cov = np.array(self.cov, dtype=float, copy=True)
        if self.cov.ndim != 2 or self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError(f"cov must be square, got shape {self.cov.shape}")
        allowed = {(reduced_size, reduced_size), (self.augmented_size, self.augmented_size)}
        if self.cov.shape not in allowed:
            raise ValueError(
                "cov must use reduced- or full-quaternion coordinates; "
                f"expected one of {sorted(allowed)}, got {self.cov.shape}"
            )

        if self.int_cov is None:
            self.int_cov = np.zeros_like(self.cov)
        else:
            self.int_cov = np.array(self.int_cov, dtype=float, copy=True)
        if self.int_cov.shape != self.cov.shape:
            raise ValueError(
                f"int_cov must match cov shape {self.cov.shape}, got {self.int_cov.shape}"
            )

    @property
    def augmented_size(self) -> int:
        return 7 + self.h.size + self.act_bias.size + self.sens_bias.size + self.dist_param.size

    @property
    def uses_reduced_quaternion_covariance(self) -> bool:
        return self.cov.shape == (self.augmented_size - 1, self.augmented_size - 1)

    @classmethod
    def from_estimator_array(
        cls,
        value: Any,
        *,
        n_rw: int = 0,
        n_act_bias: int = 0,
        n_sens_bias: int = 0,
        n_dist_param: int = 0,
        cov: Any = None,
        int_cov: Any = None,
    ) -> EstimatedState:
        array = _vector(value, name="estimated state")
        lengths = (n_rw, n_act_bias, n_sens_bias, n_dist_param)
        if any(length < 0 for length in lengths):
            raise ValueError("estimated-state block lengths cannot be negative")
        expected = 7 + sum(lengths)
        if array.size != expected:
            raise ValueError(f"estimated state must have length {expected}, got {array.size}")
        i = 7
        h = array[i : i + n_rw]
        i += n_rw
        act_bias = array[i : i + n_act_bias]
        i += n_act_bias
        sens_bias = array[i : i + n_sens_bias]
        i += n_sens_bias
        dist_param = array[i : i + n_dist_param]
        return cls(
            w=array[:3],
            q=array[3:7],
            h=h,
            act_bias=act_bias,
            sens_bias=sens_bias,
            dist_param=dist_param,
            cov=cov,
            int_cov=int_cov,
        )

    def as_estimator_array(self) -> np.ndarray:
        return np.concatenate(
            (self.w, self.q, self.h, self.act_bias, self.sens_bias, self.dist_param)
        )

    def copy(self) -> EstimatedState:
        return EstimatedState(
            w=self.w,
            q=self.q,
            h=self.h,
            act_bias=self.act_bias,
            sens_bias=self.sens_bias,
            dist_param=self.dist_param,
            cov=self.cov,
            int_cov=self.int_cov,
        )

    def normalized(self) -> EstimatedState:
        norm = float(np.linalg.norm(self.q))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("q must have a finite, non-zero norm to normalize")
        result = self.copy()
        result.q = result.q / norm
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "w": self.w.tolist(),
            "q": self.q.tolist(),
            "h": self.h.tolist(),
            "act_bias": self.act_bias.tolist(),
            "sens_bias": self.sens_bias.tolist(),
            "dist_param": self.dist_param.tolist(),
            "cov": self.cov.tolist(),
            "int_cov": self.int_cov.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EstimatedState:
        return cls(
            w=payload["w"],
            q=payload["q"],
            h=payload.get("h", ()),
            act_bias=payload.get("act_bias", ()),
            sens_bias=payload.get("sens_bias", ()),
            dist_param=payload.get("dist_param", ()),
            cov=payload.get("cov"),
            int_cov=payload.get("int_cov"),
        )
