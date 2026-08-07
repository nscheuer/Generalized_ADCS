"""Typed spacecraft attitude-state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


# These duplicate quat_mult/quat_inv from ADCS.helpers.math_helpers, which
# cannot be imported here: math_helpers -> ADCS.orbits.universal_constants
# -> ADCS.orbits.__init__ -> orbit_factory -> orbital_state -> ADCS.state is
# a circular import. This module must stay an import leaf.
def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p0, pv = p[0], p[1:]
    q0, qv = q[0], q[1:]
    return np.concatenate(([p0 * q0 - pv @ qv], p0 * qv + q0 * pv + np.cross(pv, qv)))


def _interpolate_quat(q0: np.ndarray, q1: np.ndarray, alpha: float, method: str) -> np.ndarray:
    # q and -q represent the same rotation: flip to the shortest arc first, so
    # antipodal representations never interpolate through the origin.
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if method == "nlerp" or dot > 1.0 - 1e-9:
        blended = (1.0 - alpha) * q0 + alpha * q1
    elif method == "slerp":
        theta = np.arccos(min(dot, 1.0))
        blended = (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
    else:
        raise ValueError(f"method must be 'slerp' or 'nlerp', got {method!r}")
    norm = float(np.linalg.norm(blended))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("interpolated quaternion has zero or non-finite norm")
    return blended / norm


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

    def interpolate(self, other: State, alpha: float, *, method: str = "slerp") -> State:
        """Blend two states: linear on ``w``/``h``, SLERP or NLERP on ``q``.

        Both methods are shortest-arc (sign-corrected), so antipodal quaternion
        representations of nearby rotations interpolate correctly. ``alpha`` is
        not clamped; values outside ``[0, 1]`` extrapolate.
        """
        if not isinstance(other, State):
            raise TypeError(f"other must be a State, got {type(other).__name__}")
        if self.h.size != other.h.size:
            raise ValueError("states must have the same number of reaction-wheel states")
        alpha = float(alpha)
        return State(
            w=(1.0 - alpha) * self.w + alpha * other.w,
            q=_interpolate_quat(self.q, other.q, alpha, method),
            h=(1.0 - alpha) * self.h + alpha * other.h,
        )

    def subtract(self, ref: State) -> np.ndarray:
        """Manifold difference ``self ⊖ ref`` as a reduced error vector.

        Returns ``[w - w_ref, 2·vec(q_ref⁻¹ ⊗ q), h - h_ref]`` (length
        ``6 + n_rw``), with the error quaternion sign-corrected to the shortest
        rotation so the attitude block is exactly inverted by
        :meth:`add_error`. Note this is the plain quaternion-vector convention,
        not the 2×MRP convention used by the TVLQR tracker.
        """
        if not isinstance(ref, State):
            raise TypeError(f"ref must be a State, got {type(ref).__name__}")
        if self.h.size != ref.h.size:
            raise ValueError("states must have the same number of reaction-wheel states")
        dq = _quat_product(_quat_conjugate(ref.q), self.q)
        dq = dq / np.linalg.norm(dq)
        if dq[0] < 0.0:
            dq = -dq
        return np.concatenate((self.w - ref.w, 2.0 * dq[1:], self.h - ref.h))

    def add_error(self, delta: np.ndarray) -> State:
        """Apply a reduced error vector (the inverse of :meth:`subtract`).

        ``delta`` is ``[δw, δθ, δh]`` of length ``6 + n_rw``; the attitude block
        ``δθ = 2·vec(dq)`` is retracted onto the quaternion manifold via
        ``q ⊗ dq`` with ``dq = [√(1−|δθ/2|²), δθ/2]``.
        """
        delta = _vector(delta, name="delta", size=6 + self.h.size)
        v = delta[3:6] / 2.0
        vv = float(v @ v)
        if vv > 1.0:
            raise ValueError("attitude error block exceeds the unit-quaternion range (|δθ| > 2)")
        dq = np.concatenate(([np.sqrt(1.0 - vv)], v))
        q = _quat_product(self.q, dq)
        return State(w=self.w + delta[:3], q=q / np.linalg.norm(q), h=self.h + delta[6:])

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
class EstimatorState(State):
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
    ) -> EstimatorState:
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

    def copy(self) -> EstimatorState:
        return EstimatorState(
            w=self.w,
            q=self.q,
            h=self.h,
            act_bias=self.act_bias,
            sens_bias=self.sens_bias,
            dist_param=self.dist_param,
            cov=self.cov,
            int_cov=self.int_cov,
        )

    def normalized(self) -> EstimatorState:
        norm = float(np.linalg.norm(self.q))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("q must have a finite, non-zero norm to normalize")
        result = self.copy()
        result.q = result.q / norm
        return result

    def interpolate(self, other: State, alpha: float, *, method: str = "slerp") -> EstimatorState:
        """Blend two estimated states (see :meth:`State.interpolate`).

        Bias and disturbance blocks interpolate linearly. Covariances also
        blend linearly — a convex combination of PSD matrices stays PSD, but
        this is a convenience for plotting/resampling, not a geodesic
        covariance interpolation.
        """
        if not isinstance(other, EstimatorState):
            raise TypeError(f"other must be an EstimatorState, got {type(other).__name__}")
        blocks = ("h", "act_bias", "sens_bias", "dist_param")
        for name in blocks:
            if getattr(self, name).size != getattr(other, name).size:
                raise ValueError(f"states must have matching {name} sizes to interpolate")
        if self.cov.shape != other.cov.shape:
            raise ValueError("states must use the same covariance convention to interpolate")
        alpha = float(alpha)

        def lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (1.0 - alpha) * a + alpha * b

        return EstimatorState(
            w=lerp(self.w, other.w),
            q=_interpolate_quat(self.q, other.q, alpha, method),
            h=lerp(self.h, other.h),
            act_bias=lerp(self.act_bias, other.act_bias),
            sens_bias=lerp(self.sens_bias, other.sens_bias),
            dist_param=lerp(self.dist_param, other.dist_param),
            cov=lerp(self.cov, other.cov),
            int_cov=lerp(self.int_cov, other.int_cov),
        )

    def subtract(self, ref: State) -> np.ndarray:
        """Manifold difference including bias/disturbance blocks.

        Returns ``[δw, 2·vec(q_ref⁻¹ ⊗ q), δh, δact_bias, δsens_bias,
        δdist_param]`` (length ``augmented_size − 1``); exact inverse of
        :meth:`add_error`.
        """
        if not isinstance(ref, EstimatorState):
            raise TypeError(f"ref must be an EstimatorState, got {type(ref).__name__}")
        for name in ("act_bias", "sens_bias", "dist_param"):
            if getattr(self, name).size != getattr(ref, name).size:
                raise ValueError(f"states must have matching {name} sizes to subtract")
        base = State.subtract(self, ref)
        return np.concatenate(
            (
                base,
                self.act_bias - ref.act_bias,
                self.sens_bias - ref.sens_bias,
                self.dist_param - ref.dist_param,
            )
        )

    def add_error(self, delta: np.ndarray) -> EstimatorState:
        """Apply a reduced error vector of length ``augmented_size − 1``.

        The attitude block is retracted as in :meth:`State.add_error`; all
        other blocks add linearly. Covariances are carried over unchanged.
        """
        delta = _vector(delta, name="delta", size=self.augmented_size - 1)
        base = State.add_error(self, delta[: 6 + self.h.size])
        i = 6 + self.h.size
        result = self.copy()
        result.w = base.w
        result.q = base.q
        result.h = base.h
        result.act_bias = self.act_bias + delta[i : i + self.act_bias.size]
        i += self.act_bias.size
        result.sens_bias = self.sens_bias + delta[i : i + self.sens_bias.size]
        i += self.sens_bias.size
        result.dist_param = self.dist_param + delta[i:]
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
    def from_dict(cls, payload: Mapping[str, Any]) -> EstimatorState:
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
