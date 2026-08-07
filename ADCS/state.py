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


def _square_matrix(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    return array


@dataclass(slots=True, eq=False)
class State:
    """Physical spacecraft state ``[w, q, h]``.

    The class deliberately does not emulate a NumPy array. Numerical-library
    boundaries must use :meth:`as_array` explicitly.
    """

    w: np.ndarray
    q: np.ndarray
    h: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "w":
            value = _vector(value, name=name, size=3)
        elif name == "q":
            value = _vector(value, name=name, size=4)
        elif name == "h":
            value = _vector(value, name=name)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.w = _vector(self.w, name="w", size=3)
        self.q = _vector(self.q, name="q", size=4)
        self.h = _vector(self.h, name="h")

    def __eq__(self, other: object) -> bool:
        if type(other) is not State:
            return NotImplemented
        return (
            np.array_equal(self.w, other.w)
            and np.array_equal(self.q, other.q)
            and np.array_equal(self.h, other.h)
        )

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
        from ADCS.helpers.math_helpers import interpolate_quat

        return State(
            w=(1.0 - alpha) * self.w + alpha * other.w,
            q=interpolate_quat(self.q, other.q, alpha, method),
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
        from ADCS.helpers.math_helpers import quat_inv, quat_mult

        dq = quat_mult(quat_inv(ref.q), self.q)
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
        from ADCS.helpers.math_helpers import quat_mult

        q = quat_mult(self.q, dq)
        return State(w=self.w + delta[:3], q=q / np.linalg.norm(q), h=self.h + delta[6:])

    def to_dict(self) -> dict[str, Any]:
        return {"w": self.w.tolist(), "q": self.q.tolist(), "h": self.h.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> State:
        return cls(w=payload["w"], q=payload["q"], h=payload.get("h", ()))

    @staticmethod
    def stack(states: Iterable[State]) -> np.ndarray:
        states = list(states)
        bad = [type(state).__name__ for state in states if not isinstance(state, State)]
        if bad:
            raise TypeError(f"State.stack expects State objects, got {bad[0]}")
        rows = [state.as_array() for state in states]
        if not rows:
            return np.empty((0, 0), dtype=float)
        widths = {row.size for row in rows}
        if len(widths) != 1:
            raise ValueError("all states must have the same number of reaction-wheel states")
        return np.vstack(rows)


@dataclass(slots=True, eq=False)
class EstimatorState(State):
    """Physical state plus estimated parameters and uncertainty."""

    act_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    sens_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    dist_param: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    cov: np.ndarray | None = None
    int_cov: np.ndarray | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        vector_sizes = {
            "w": 3,
            "q": 4,
            "h": None,
            "act_bias": None,
            "sens_bias": None,
            "dist_param": None,
        }
        if name in vector_sizes:
            value = _vector(value, name=name, size=vector_sizes[name])
            self._validate_existing_covariances_for_vector_assignment(name, value)
        elif name in ("cov", "int_cov"):
            value = _square_matrix(value, name=name)
            self._validate_covariance_assignment(name, value)
        object.__setattr__(self, name, value)

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

    def _allowed_covariance_shapes(
        self,
        *,
        replacing_name: str | None = None,
        replacing_value: np.ndarray | None = None,
    ) -> set[tuple[int, int]] | None:
        sizes = {}
        for name in ("h", "act_bias", "sens_bias", "dist_param"):
            if replacing_name == name:
                sizes[name] = replacing_value.size
                continue
            try:
                sizes[name] = getattr(self, name).size
            except AttributeError:
                return None
        augmented_size = (
            7 + sizes["h"] + sizes["act_bias"] + sizes["sens_bias"] + sizes["dist_param"]
        )
        return {(augmented_size - 1, augmented_size - 1), (augmented_size, augmented_size)}

    def _validate_existing_covariances_for_vector_assignment(
        self,
        name: str,
        value: np.ndarray,
    ) -> None:
        allowed = self._allowed_covariance_shapes(replacing_name=name, replacing_value=value)
        if allowed is None:
            return
        for cov_name in ("cov", "int_cov"):
            try:
                cov = getattr(self, cov_name)
            except AttributeError:
                continue
            if cov is not None and cov.shape not in allowed:
                raise ValueError(
                    f"{name} assignment would make {cov_name} shape {cov.shape} "
                    f"incompatible with expected covariance shapes {sorted(allowed)}"
                )

    def _validate_covariance_assignment(
        self,
        name: str,
        value: np.ndarray | None,
    ) -> None:
        if value is None:
            return
        allowed = self._allowed_covariance_shapes()
        if allowed is not None and value.shape not in allowed:
            raise ValueError(
                f"{name} must use reduced- or full-quaternion coordinates; "
                f"expected one of {sorted(allowed)}, got {value.shape}"
            )
        other_name = "int_cov" if name == "cov" else "cov"
        try:
            other = getattr(self, other_name)
        except AttributeError:
            return
        if other is not None and other.shape != value.shape:
            raise ValueError(
                f"{name} must match {other_name} shape {other.shape}, got {value.shape}"
            )

    def __eq__(self, other: object) -> bool:
        if type(other) is not EstimatorState:
            return NotImplemented
        return (
            np.array_equal(self.w, other.w)
            and np.array_equal(self.q, other.q)
            and np.array_equal(self.h, other.h)
            and np.array_equal(self.act_bias, other.act_bias)
            and np.array_equal(self.sens_bias, other.sens_bias)
            and np.array_equal(self.dist_param, other.dist_param)
            and np.array_equal(self.cov, other.cov)
            and np.array_equal(self.int_cov, other.int_cov)
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
        from ADCS.helpers.math_helpers import interpolate_quat

        def lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (1.0 - alpha) * a + alpha * b

        return EstimatorState(
            w=lerp(self.w, other.w),
            q=interpolate_quat(self.q, other.q, alpha, method),
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
