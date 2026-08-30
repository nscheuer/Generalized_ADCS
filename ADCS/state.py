"""Typed spacecraft attitude-state containers."""

from __future__ import annotations

__all__ = ["State", "EstimatorState"]

from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Literal, Mapping

import numpy as np

from ADCS.covariance import Covariance


QuaternionMode = Literal[
    "quaternion_vector",
    "rotation_vector",
    "mrp",
    "two_mrp",
    "cayley",
    "full_quaternion",
]
QuaternionOrder = Literal["right", "left"]
StateCoordinates = Literal["full", "tangent"]


def _quaternion_mode(value: str) -> QuaternionMode:
    aliases = {
        "quat_vec": "quaternion_vector",
        "rotvec": "rotation_vector",
        "2mrp": "two_mrp",
    }
    value = aliases.get(value, value)
    allowed = {
        "quaternion_vector",
        "rotation_vector",
        "mrp",
        "two_mrp",
        "cayley",
        "full_quaternion",
    }
    if value not in allowed:
        raise ValueError(
            f"unsupported quaternion mode {value!r}; expected one of {sorted(allowed)}"
        )
    return value  # type: ignore[return-value]


def _quaternion_order(value: str) -> QuaternionOrder:
    if value not in ("right", "left"):
        raise ValueError(f"quaternion order must be 'right' or 'left', got {value!r}")
    return value  # type: ignore[return-value]


def _unit_quaternion(value: Any, *, name: str = "q") -> np.ndarray:
    q = _vector(value, name=name, size=4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must have a finite, non-zero norm")
    return q / norm


def _quat_delta_from_vector(value: Any, mode: str) -> np.ndarray:
    vector = _vector(value, name="attitude delta", size=3)
    mode = _quaternion_mode(mode)
    if mode == "full_quaternion":
        raise ValueError("full_quaternion uses a four-element additive attitude block")
    if mode == "quaternion_vector":
        qv = vector / 2.0
        qv_norm_sq = float(qv @ qv)
        if qv_norm_sq > 1.0:
            raise ValueError("attitude delta exceeds quaternion-vector range (norm > 2)")
        return np.concatenate(([np.sqrt(max(0.0, 1.0 - qv_norm_sq))], qv))

    from ADCS.helpers.math_helpers import rot_exp, vec3_to_quat

    if mode == "rotation_vector":
        return rot_exp(vector)
    helper_mode = {"mrp": 1, "two_mrp": 6, "cayley": 2}[mode]
    return vec3_to_quat(vector, helper_mode)


def _quat_delta_to_vector(value: Any, mode: str, *, shortest: bool) -> np.ndarray:
    q = _unit_quaternion(value, name="quaternion delta")
    mode = _quaternion_mode(mode)
    if mode == "full_quaternion":
        raise ValueError("full_quaternion does not convert to a three-element attitude vector")
    if shortest and q[0] < 0.0:
        q = -q
    if mode == "quaternion_vector":
        return 2.0 * q[1:]
    if mode == "rotation_vector":
        vector_norm = float(np.linalg.norm(q[1:]))
        if vector_norm < 1e-15:
            return 2.0 * q[1:]
        return (2.0 * np.arctan2(vector_norm, q[0]) / vector_norm) * q[1:]

    from ADCS.helpers.math_helpers import quat_to_vec3

    helper_mode = {"mrp": 1, "two_mrp": 5, "cayley": 2}[mode]
    return quat_to_vec3(q, helper_mode)


def _quaternion_tangent_scale(mode: str) -> float:
    mode = _quaternion_mode(mode)
    if mode == "full_quaternion":
        raise ValueError("full_quaternion has no reduced quaternion tangent scale")
    return {
        "quaternion_vector": 0.5,
        "rotation_vector": 0.5,
        "mrp": 2.0,
        "two_mrp": 1.0,
        "cayley": 1.0,
    }[mode]


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def _rotation_vector_reset_jacobian(vector: np.ndarray, order: QuaternionOrder) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    theta_sq = theta * theta
    if theta < 1.0e-8:
        a = 0.5 - theta_sq / 24.0
        b = 1.0 / 6.0 - theta_sq / 120.0
    else:
        a = (1.0 - np.cos(theta)) / theta_sq
        b = (theta - np.sin(theta)) / (theta_sq * theta)
    cross = _skew(vector)
    sign = -1.0 if order == "right" else 1.0
    return np.eye(3) + sign * a * cross + b * (cross @ cross)


def _quaternion_vector_reset_jacobian(vector: np.ndarray, order: QuaternionOrder) -> np.ndarray | None:
    qv = vector / 2.0
    qv_norm_sq = float(qv @ qv)
    if qv_norm_sq > 1.0:
        raise ValueError("attitude delta exceeds quaternion-vector range (norm > 2)")
    q0 = np.sqrt(max(0.0, 1.0 - qv_norm_sq))
    if q0 <= 1.0e-12:
        return None
    sign = -1.0 if order == "right" else 1.0
    return q0 * np.eye(3) + np.outer(qv, qv) / q0 + sign * _skew(qv)


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


@dataclass(slots=True, eq=False, init=False)
class State:
    r"""Physical spacecraft state :math:`x=[\boldsymbol\omega,\mathbf q,\mathbf h]`.

    .. math::

        x \in \mathbb R^3 \times \mathbb S^3 \times \mathbb R^{n_h},
        \qquad
        \delta x =
        \begin{bmatrix}
        \delta\boldsymbol\omega &
        \delta\boldsymbol\theta &
        \delta\mathbf h
        \end{bmatrix}^{T}
        \in\mathbb R^{6+n_h}.

    The unit quaternion has the double-cover equivalence
    :math:`\mathbf q\sim-\mathbf q`. Attitude differences therefore live in
    the three-dimensional tangent block :math:`\delta\boldsymbol\theta`, not
    in the four stored quaternion coefficients. For a relative quaternion
    :math:`\delta\mathbf q=[\eta,\boldsymbol\epsilon]`, two common coordinate
    maps supported by :meth:`~ADCS.state.State.minus` are

    .. math::

        \phi_{qv}(\delta\mathbf q)=2\boldsymbol\epsilon,
        \qquad
        \phi_{rv}(\delta\mathbf q)=
        2\operatorname{atan2}(\lVert\boldsymbol\epsilon\rVert,\eta)
        \frac{\boldsymbol\epsilon}{\lVert\boldsymbol\epsilon\rVert}.

    The class deliberately does not emulate a NumPy array. Numerical-library
    boundaries use :meth:`~ADCS.state.State.as_array` explicitly. Estimated
    parameters and covariance are provided by :class:`~ADCS.state.EstimatorState`.
    """

    w: np.ndarray
    q: np.ndarray
    h: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    _slice_cache: dict[StateCoordinates, dict[str, slice]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    DEFAULT_QUATERNION_MODE: ClassVar[QuaternionMode] = "quaternion_vector"
    DEFAULT_QUATERNION_ORDER: ClassVar[QuaternionOrder] = "right"
    _BLOCK_FIELDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("angular_velocity", "w"),
        ("attitude", "q"),
        ("wheel_momentum", "h"),
    )
    _BLOCK_NAMES: ClassVar[tuple[str, ...]] = tuple(name for name, _ in _BLOCK_FIELDS)
    _BLOCK_ATTRIBUTES: ClassVar[Mapping[str, str]] = dict(_BLOCK_FIELDS)
    _BLOCK_ALIASES: ClassVar[Mapping[str, str]] = {
        "quaternion": "attitude",
        "estimated_parameter": "estimated_parameters",
    }

    def __init__(self, w: Any, q: Any, h: Any = ()) -> None:
        object.__setattr__(self, "_slice_cache", {})
        self.w = w
        self.q = q
        self.h = h

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "w":
            value = _vector(value, name=name, size=3)
        elif name == "q":
            value = _vector(value, name=name, size=4)
        elif name == "h":
            value = _vector(value, name=name)
        object.__setattr__(self, name, value)
        if name in self._BLOCK_ATTRIBUTES.values() and hasattr(self, "_slice_cache"):
            self._slice_cache.clear()

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

    @property
    def block_names(self) -> tuple[str, ...]:
        """Canonical state-block names in storage order."""
        return self._BLOCK_NAMES

    def block(self, name: str) -> np.ndarray:
        """Return a state block by semantic name without copying it.

        This is the bridge between estimator mathematics and storage names:
        ``state.block("attitude")`` returns ``state.q`` and
        ``state.block("wheel_momentum")`` returns ``state.h``.  Callers that
        need ownership should copy the returned array.
        """
        canonical = self._BLOCK_ALIASES.get(name, name)
        try:
            attribute = self._BLOCK_ATTRIBUTES[canonical]
        except KeyError:
            raise KeyError(
                f"unknown state block {name!r}; expected one of {self.block_names}"
            ) from None
        return getattr(self, attribute)

    @staticmethod
    def _coordinates(value: str) -> StateCoordinates:
        if value not in ("full", "tangent"):
            raise ValueError(f"coordinates must be 'full' or 'tangent', got {value!r}")
        return value  # type: ignore[return-value]

    def block_size(self, name: str, *, coordinates: str = "full") -> int:
        """Return one block's dimension in stored or local coordinates."""
        coordinates = self._coordinates(coordinates)
        canonical = self._BLOCK_ALIASES.get(name, name)
        if canonical == "physical":
            return sum(
                self._block_width(block_name, attribute, coordinates)
                for block_name, attribute in State._BLOCK_FIELDS
            )
        if canonical == "estimated_parameters":
            return sum(
                self._block_width(block_name, attribute, coordinates)
                for block_name, attribute in self._BLOCK_FIELDS[len(State._BLOCK_FIELDS) :]
            )
        try:
            attribute = self._BLOCK_ATTRIBUTES[canonical]
        except KeyError:
            raise KeyError(
                f"unknown state block {name!r}; expected one of {self.block_names}"
            ) from None
        return self._block_width(canonical, attribute, coordinates)

    def _block_width(
        self,
        name: str,
        attribute: str,
        coordinates: StateCoordinates,
    ) -> int:
        if coordinates == "tangent" and name == "attitude":
            return 3
        return getattr(self, attribute).size

    def slice(self, name: str, *, coordinates: str = "full") -> slice:
        """Return a block slice in stored or local state coordinates.

        ``physical`` selects angular velocity through wheel momentum;
        ``estimated_parameters`` selects all augmented bias/parameter blocks.
        ``quaternion`` remains an alias for the canonical ``attitude`` name.
        """
        coordinates = self._coordinates(coordinates)
        cached = self._cached_slices(coordinates)
        canonical = self._BLOCK_ALIASES.get(name, name)
        if canonical == "physical":
            return cached["physical"]
        if canonical == "estimated_parameters":
            return cached["estimated_parameters"]
        if canonical in cached:
            return cached[canonical]
        raise KeyError(f"unknown state block {name!r}; expected one of {self.block_names}")

    def _cached_slices(self, coordinates: StateCoordinates) -> dict[str, slice]:
        cached = self._slice_cache.get(coordinates)
        if cached is not None:
            return cached
        result: dict[str, slice] = {}
        start = 0
        physical_stop = 0
        for index, (name, attribute) in enumerate(self._BLOCK_FIELDS):
            width = self._block_width(name, attribute, coordinates)
            result[name] = slice(start, start + width)
            start += width
            if index + 1 == len(State._BLOCK_FIELDS):
                physical_stop = start
        result["physical"] = slice(0, physical_stop)
        result["estimated_parameters"] = slice(physical_stop, start)
        if coordinates == "full":
            result["quaternion"] = result["attitude"]
        self._slice_cache[coordinates] = result
        return result

    def slices(self, *, coordinates: str = "full") -> dict[str, slice]:
        """Return all named slices for one coordinate representation.

        For repeated block assembly, obtain this mapping once and reuse it.
        Single-block callers should prefer :meth:`slice`, which avoids
        allocating a mapping.
        """
        coordinates = self._coordinates(coordinates)
        return dict(self._cached_slices(coordinates))

    @property
    def full_slices(self) -> dict[str, slice]:
        """Compatibility mapping of stored-vector slices.

        New code should prefer :meth:`slice` or :meth:`slices`; this property
        retains the established ``quaternion`` key and compact mapping.
        """
        result = self.slices(coordinates="full")
        result.pop("attitude")
        result.pop("physical")
        result.pop("estimated_parameters")
        return result

    @property
    def tangent_slices(self) -> dict[str, slice]:
        """Compatibility mapping of local-error slices."""
        result = self.slices(coordinates="tangent")
        result.pop("estimated_parameters")
        return result

    def size(self, *, coordinates: str = "full") -> int:
        """Return the total stored or local state dimension."""
        coordinates = self._coordinates(coordinates)
        return self._cached_slices(coordinates)["estimated_parameters"].stop

    def validate_layout(self, **expected_sizes: int) -> None:
        """Validate selected block dimensions with concise diagnostics.

        Example: ``state.validate_layout(wheel_momentum=3, sensor_bias=6)``.
        Dimensions refer to stored coordinates; only attitude differs in local
        coordinates, and hardware-facing layout checks should never depend on
        the chosen attitude chart.
        """
        for name, expected in expected_sizes.items():
            if not isinstance(expected, (int, np.integer)) or expected < 0:
                raise ValueError(f"expected size for {name!r} must be a non-negative integer")
            actual = self.block_size(name)
            if actual != expected:
                raise ValueError(
                    f"state block {name!r} has size {actual}, expected {expected}"
                )

    @property
    def full_size(self) -> int:
        """Number of stored scalar state elements, including all four quaternion elements."""
        return self.size(coordinates="full")

    @property
    def tangent_size(self) -> int:
        """Dimension of the local state coordinates, with three attitude elements."""
        return self.size(coordinates="tangent")

    @property
    def error_size(self) -> int:
        """Alias for :attr:`~ADCS.state.State.tangent_size`."""
        return self.tangent_size

    def copy(self) -> State:
        result = object.__new__(State)
        object.__setattr__(result, "_slice_cache", {})
        object.__setattr__(result, "w", self.w.copy())
        object.__setattr__(result, "q", self.q.copy())
        object.__setattr__(result, "h", self.h.copy())
        return result

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

    @staticmethod
    def quaternion_delta_from_vector(
        value: Any,
        *,
        mode: str = DEFAULT_QUATERNION_MODE,
    ) -> np.ndarray:
        """Convert three local attitude coordinates into a unit quaternion delta."""
        return _quat_delta_from_vector(value, mode)

    @staticmethod
    def quaternion_delta_to_vector(
        value: Any,
        *,
        mode: str = DEFAULT_QUATERNION_MODE,
        shortest: bool = True,
    ) -> np.ndarray:
        """Convert a quaternion delta into three local attitude coordinates."""
        return _quat_delta_to_vector(value, mode, shortest=shortest)

    def aligned_quaternion(self, reference: Any) -> np.ndarray:
        """Return this state's unit quaternion with the sign nearest ``reference``."""
        q = _unit_quaternion(self.q)
        ref = _unit_quaternion(reference, name="reference quaternion")
        return -q if float(q @ ref) < 0.0 else q

    def with_quaternion_delta(
        self,
        delta_q: Any,
        *,
        order: str = DEFAULT_QUATERNION_ORDER,
        normalize: bool = True,
    ) -> State:
        r"""Compose a unit delta quaternion on the right or left.

        .. math::

            \mathbf q^+ = \mathbf q\otimes\delta\mathbf q
            \quad\text{or}\quad
            \mathbf q^+ = \delta\mathbf q\otimes\mathbf q

        The Hamilton product matches
        :func:`~ADCS.helpers.math_helpers.quat_mult`.
        """
        order = _quaternion_order(order)
        delta_q = _unit_quaternion(delta_q, name="quaternion delta")
        from ADCS.helpers.math_helpers import quat_mult

        q = quat_mult(self.q, delta_q) if order == "right" else quat_mult(delta_q, self.q)
        if normalize:
            q = _unit_quaternion(q)
        result = self.copy()
        result.q = q
        return result

    def minus(
        self,
        ref: State,
        *,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
        shortest: bool = True,
    ) -> np.ndarray:
        r"""Return :math:`x\boxminus x_{\mathrm{ref}}` in local coordinates.

        For right errors,

        .. math::

            \delta\mathbf q = \mathbf q_{\mathrm{ref}}^{-1}\otimes\mathbf q,
            \qquad
            \delta x = [\Delta\boldsymbol\omega,
            \phi(\delta\mathbf q),\Delta\mathbf h].

        Left errors reverse the quaternion product. See
        :meth:`~ADCS.state.State.plus` for the inverse operation.
        """
        if not isinstance(ref, State):
            raise TypeError(f"ref must be a State, got {type(ref).__name__}")
        if self.h.size != ref.h.size:
            raise ValueError("states must have the same number of reaction-wheel states")
        quaternion_mode = _quaternion_mode(quaternion_mode)
        if quaternion_mode == "full_quaternion":
            q = self.aligned_quaternion(ref.q) if shortest else _unit_quaternion(self.q)
            return np.concatenate((self.w - ref.w, q - ref.q, self.h - ref.h))
        from ADCS.helpers.math_helpers import quat_inv, quat_mult

        order = _quaternion_order(quaternion_order)
        if order == "right":
            dq = quat_mult(quat_inv(ref.q), self.q)
        else:
            dq = quat_mult(self.q, quat_inv(ref.q))
        attitude = _quat_delta_to_vector(dq, quaternion_mode, shortest=shortest)
        return np.concatenate((self.w - ref.w, attitude, self.h - ref.h))

    def plus(
        self,
        delta: Any,
        *,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
        normalize: bool = True,
    ) -> State:
        r"""Return :math:`x\boxplus\delta x`.

        For the default right-error convention,

        .. math::

            x\boxplus\delta x =
            [\boldsymbol\omega+\delta\boldsymbol\omega,
            \mathbf q\otimes\phi^{-1}(\delta\boldsymbol\theta),
            \mathbf h+\delta\mathbf h].

        ``full_quaternion`` uses additive four-element quaternion coordinates
        followed by normalization. Other modes use three attitude coordinates.
        This is the inverse of :meth:`~ADCS.state.State.minus` locally.
        """
        quaternion_mode = _quaternion_mode(quaternion_mode)
        if quaternion_mode == "full_quaternion":
            delta = _vector(delta, name="delta", size=self.full_size)
            slices = self.slices(coordinates="full")
            result = self.copy()
            result.w = self.w + delta[slices["angular_velocity"]]
            q = self.q + delta[slices["attitude"]]
            result.q = _unit_quaternion(q) if normalize else q
            result.h = self.h + delta[slices["wheel_momentum"]]
            return result
        delta = _vector(delta, name="delta", size=self.tangent_size)
        slices = self.slices(coordinates="tangent")
        dq = _quat_delta_from_vector(delta[slices["attitude"]], quaternion_mode)
        result = self.with_quaternion_delta(dq, order=quaternion_order, normalize=normalize)
        result.w = self.w + delta[slices["angular_velocity"]]
        result.h = self.h + delta[slices["wheel_momentum"]]
        return result

    def retract(self, delta: Any, **kwargs: Any) -> State:
        """Semantic alias for :meth:`~ADCS.state.State.plus`."""
        return self.plus(delta, **kwargs)

    def retraction_jacobian(
        self,
        delta: Any,
        *,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
        step: float = 1.0e-5,
    ) -> np.ndarray:
        r"""Return the covariance-reset Jacobian after retracting ``delta``.

        If ``updated = self.plus(delta)``, this is the differential

        .. math::

            J_{reset} = \left.\frac{\partial}{\partial\epsilon}
            \left[
            \bigl(self\boxplus(\delta+\epsilon)\bigr)
            \boxminus updated
            \right]\right|_{\epsilon=0}.

        It transports an error covariance from the old tangent point to the
        tangent point at ``updated``. Linear state blocks remain identity; only
        the attitude block is evaluated numerically. This keeps the operation
        consistent with every quaternion chart supported by :meth:`plus` and
        :meth:`minus` without imposing chart-specific formulas on filters.
        """
        quaternion_mode = _quaternion_mode(quaternion_mode)
        quaternion_order = _quaternion_order(quaternion_order)
        local_size = self.full_size if quaternion_mode == "full_quaternion" else self.tangent_size
        delta = _vector(delta, name="delta", size=local_size)
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")

        coordinates = "full" if quaternion_mode == "full_quaternion" else "tangent"
        attitude = self.slice("attitude", coordinates=coordinates)
        if quaternion_mode == "rotation_vector":
            result = np.eye(local_size)
            result[attitude, attitude] = _rotation_vector_reset_jacobian(
                delta[attitude], quaternion_order
            )
            return result
        if quaternion_mode == "quaternion_vector":
            attitude_reset = _quaternion_vector_reset_jacobian(
                delta[attitude], quaternion_order
            )
            if attitude_reset is not None:
                result = np.eye(local_size)
                result[attitude, attitude] = attitude_reset
                return result
        updated = self.plus(
            delta,
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        )
        result = np.eye(local_size)
        for column in range(attitude.start, attitude.stop):
            offset = np.zeros(local_size)
            offset[column] = step
            plus_error = self.plus(
                delta + offset,
                quaternion_mode=quaternion_mode,
                quaternion_order=quaternion_order,
            ).minus(
                updated,
                quaternion_mode=quaternion_mode,
                quaternion_order=quaternion_order,
                shortest=False,
            )
            minus_error = self.plus(
                delta - offset,
                quaternion_mode=quaternion_mode,
                quaternion_order=quaternion_order,
            ).minus(
                updated,
                quaternion_mode=quaternion_mode,
                quaternion_order=quaternion_order,
                shortest=False,
            )
            result[attitude, column] = (
                plus_error[attitude] - minus_error[attitude]
            ) / (2.0 * step)
        return result

    def transport_covariance(
        self,
        covariance: Covariance,
        delta: Any,
        **retraction_kwargs: Any,
    ) -> Covariance:
        r"""Transport covariance after ``self.plus(delta)``.

        Given an update covariance expressed around ``self``, return

        .. math::

            P_{new}=J_{reset}P J_{reset}^{T}

        expressed around the retracted state. The covariance representation
        and PSD policy are preserved by :meth:`Covariance.transformed`.
        """
        if not isinstance(covariance, Covariance):
            raise TypeError(
                f"covariance must be a Covariance, got {type(covariance).__name__}"
            )
        reset = self.retraction_jacobian(delta, **retraction_kwargs)
        return covariance.transformed(reset)

    def local_coordinates(self, ref: State, **kwargs: Any) -> np.ndarray:
        """Semantic alias for :meth:`~ADCS.state.State.minus`."""
        return self.minus(ref, **kwargs)

    def subtract(self, ref: State) -> np.ndarray:
        """Compatibility wrapper for :meth:`~ADCS.state.State.minus`."""
        return self.minus(ref)

    def add_error(self, delta: np.ndarray) -> State:
        """Compatibility wrapper for :meth:`~ADCS.state.State.plus`."""
        return self.plus(delta)

    def tangent_map(
        self,
        *,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
    ) -> np.ndarray:
        r"""Return the local-to-full differential map :math:`G(x)`.

        .. math::

            G(x)=\operatorname{diag}(I_3,sW_{\pm}(\mathbf q),I_{n_h}),
            \qquad
            W_{\pm}(\mathbf q)=
            \begin{bmatrix}-\mathbf q_v^T\\q_0I_3\pm[\mathbf q_v]_\times\end{bmatrix}.

        Thus a local perturbation and its first-order full-state displacement
        are related by

        .. math::

            (x\boxplus\delta x)-x = G(x)\,\delta x
            +\mathcal O(\lVert\delta x\rVert^2).

        The sign is positive for right errors and negative for left errors.
        """
        quaternion_mode = _quaternion_mode(quaternion_mode)
        if quaternion_mode == "full_quaternion":
            return self.normalization_jacobian()
        order = _quaternion_order(quaternion_order)
        q = _unit_quaternion(self.q)
        q0, qv = q[0], q[1:]
        cross = _skew(qv)
        sign = 1.0 if order == "right" else -1.0
        quaternion_block = _quaternion_tangent_scale(quaternion_mode) * np.vstack(
            (-qv, q0 * np.eye(3) + sign * cross)
        )
        full = self._cached_slices("full")
        tangent = self._cached_slices("tangent")
        result = np.zeros((self.full_size, self.tangent_size), dtype=float)
        for name in self.block_names:
            if name == "attitude":
                result[full[name], tangent[name]] = quaternion_block
            else:
                result[full[name], tangent[name]] = np.eye(self.block_size(name))
        return result

    def tangent_pinv(
        self,
        *,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
    ) -> np.ndarray:
        r"""Return :math:`G(x)^\dagger`, the analytical pseudoinverse of
        :meth:`~ADCS.state.State.tangent_map`.

        For reduced attitude coordinates the quaternion block satisfies

        .. math::

            (sW_\pm)^\dagger=\frac{1}{s}W_\pm^T,
            \qquad
            G^\dagger G=I,
            \qquad
            GG^\dagger=\Pi_{T_x\mathcal M},

        where :math:`\Pi_{T_x\mathcal M}` projects a full quaternion
        displacement onto the unit-quaternion tangent space.
        """
        quaternion_mode = _quaternion_mode(quaternion_mode)
        if quaternion_mode == "full_quaternion":
            return np.linalg.pinv(self.normalization_jacobian())
        tangent = self.tangent_map(
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
        )
        scale = _quaternion_tangent_scale(quaternion_mode)
        result = tangent.T.copy()
        result[
            self.slice("attitude", coordinates="tangent"),
            self.slice("attitude", coordinates="full"),
        ] /= scale**2
        return result

    def normalization_jacobian(self) -> np.ndarray:
        r"""Return the full-state quaternion-normalization Jacobian.

        .. math::

            N_q=\frac{1}{\lVert\mathbf q\rVert}
            \left(I_4-\frac{\mathbf q\mathbf q^T}{\lVert\mathbf q\rVert^2}\right).
        """
        q = self.q
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("q must have a finite, non-zero norm")
        result = np.eye(self.full_size)
        attitude = self.slice("attitude", coordinates="full")
        result[attitude, attitude] = np.eye(4) / norm - np.outer(q, q) / norm**3
        return result

    def is_close(self, other: State, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Compare physical states while treating ``q`` and ``-q`` as equivalent."""
        if not isinstance(other, State) or self.h.size != other.h.size:
            return False
        try:
            self_q = _unit_quaternion(self.q)
            other_q = _unit_quaternion(other.q)
        except ValueError:
            return False
        return bool(
            np.allclose(self.w, other.w, rtol=rtol, atol=atol)
            and np.allclose(self.h, other.h, rtol=rtol, atol=atol)
            and (
                np.allclose(self_q, other_q, rtol=rtol, atol=atol)
                or np.allclose(self_q, -other_q, rtol=rtol, atol=atol)
            )
        )

    @classmethod
    def mean(
        cls,
        states: Iterable[State],
        weights: Any = None,
        *,
        reference: State | None = None,
        quaternion_mode: str = DEFAULT_QUATERNION_MODE,
        quaternion_order: str = DEFAULT_QUATERNION_ORDER,
        tolerance: float = 1e-12,
        max_iterations: int = 50,
    ) -> State:
        r"""Compute the weighted manifold mean :math:`\bar x` satisfying

        .. math::

            \sum_i w_i\left(x_i\boxminus\bar x\right)=0.

        Iteration uses :meth:`~ADCS.state.State.minus` and
        :meth:`~ADCS.state.State.plus`.
        """
        values = list(states)
        if not values:
            raise ValueError("states must not be empty")
        if any(type(value) is not cls for value in values):
            raise TypeError(f"all states must be {cls.__name__} objects")
        if any(value.h.size != values[0].h.size for value in values):
            raise ValueError("states must have the same number of reaction-wheel states")
        if weights is None:
            weight_array = np.full(len(values), 1.0 / len(values))
        else:
            weight_array = _vector(weights, name="weights", size=len(values))
            total = float(np.sum(weight_array))
            if not np.isfinite(total) or abs(total) < np.finfo(float).eps:
                raise ValueError("weights must have a finite, non-zero sum")
            weight_array = weight_array / total
        current = values[0].copy() if reference is None else reference.copy()
        if type(current) is not cls:
            raise TypeError(f"reference must be a {cls.__name__}")
        local_size = (
            current.full_size
            if _quaternion_mode(quaternion_mode) == "full_quaternion"
            else current.tangent_size
        )
        for _ in range(max_iterations):
            step = sum(
                (
                    weight * value.minus(
                        current,
                        quaternion_mode=quaternion_mode,
                        quaternion_order=quaternion_order,
                    )
                    for value, weight in zip(values, weight_array)
                ),
                np.zeros(local_size),
            )
            current = current.plus(
                step,
                quaternion_mode=quaternion_mode,
                quaternion_order=quaternion_order,
            )
            if float(np.linalg.norm(step)) <= tolerance:
                return current
        raise RuntimeError(f"state mean did not converge within {max_iterations} iterations")

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


@dataclass(slots=True, eq=False, init=False)
class EstimatorState(State):
    r"""A :class:`~ADCS.state.State` with estimated parameters and uncertainty.

    The augmented estimate and its reduced local error are ordered as

    .. math::

        \hat x=
        \begin{bmatrix}
        \boldsymbol\omega & \mathbf q & \mathbf h &
        \mathbf b_a & \mathbf b_s & \mathbf d
        \end{bmatrix}^{T},
        \qquad
        P=\mathbb E\!\left[\delta x\,\delta x^T\right].

    Consequently, the default covariance has one fewer row and column than
    the stored state because :math:`\mathbf q\in\mathbb S^3` contributes only
    three local degrees of freedom. Full quaternion-coordinate covariances are
    also accepted and can be projected using
    :meth:`~ADCS.state.EstimatorState.covariance_to_reduced`.

    :attr:`covariance` and :attr:`process_noise` are authoritative
    :class:`~ADCS.covariance.Covariance` objects. The ``cov`` and ``int_cov``
    properties retain the legacy full-matrix interface during estimator
    migration.
    """

    act_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    sens_bias: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    dist_param: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    _covariance: Covariance = field(init=False, repr=False)
    _process_noise: Covariance = field(init=False, repr=False)
    _BLOCK_FIELDS: ClassVar[tuple[tuple[str, str], ...]] = State._BLOCK_FIELDS + (
        ("actuator_bias", "act_bias"),
        ("sensor_bias", "sens_bias"),
        ("disturbance_parameter", "dist_param"),
    )
    _BLOCK_NAMES: ClassVar[tuple[str, ...]] = tuple(name for name, _ in _BLOCK_FIELDS)
    _BLOCK_ATTRIBUTES: ClassVar[Mapping[str, str]] = dict(_BLOCK_FIELDS)

    def __init__(
        self,
        w: Any,
        q: Any,
        h: Any = (),
        act_bias: Any = (),
        sens_bias: Any = (),
        dist_param: Any = (),
        cov: Any = None,
        int_cov: Any = None,
        *,
        covariance: Covariance | None = None,
        process_noise: Covariance | None = None,
    ) -> None:
        object.__setattr__(self, "_slice_cache", {})
        self.w = w
        self.q = q
        self.h = h
        self.act_bias = act_bias
        self.sens_bias = sens_bias
        self.dist_param = dist_param

        if cov is not None and covariance is not None:
            raise ValueError("provide either cov or covariance, not both")
        if int_cov is not None and process_noise is not None:
            raise ValueError("provide either int_cov or process_noise, not both")

        reduced_size = self.tangent_size
        covariance_value = covariance if covariance is not None else cov
        if covariance_value is None:
            covariance_value = Covariance.zeros(
                reduced_size,
                coordinates="state_tangent",
                psd_policy="allow_indefinite",
            )
        self._covariance = self._coerce_covariance(
            covariance_value, name="cov", default_coordinates="state_tangent"
        )

        process_value = process_noise if process_noise is not None else int_cov
        if process_value is None:
            process_value = Covariance.zeros(
                self._covariance.dimension,
                form=self._covariance.form,
                coordinates=self._covariance.coordinates,
                psd_policy="allow_indefinite",
            )
        self._process_noise = self._coerce_covariance(
            process_value,
            name="int_cov",
            default_coordinates=self._covariance.coordinates,
        )
        if self._process_noise.shape != self._covariance.shape:
            raise ValueError(
                f"int_cov must match cov shape {self._covariance.shape}, "
                f"got {self._process_noise.shape}"
            )

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
        object.__setattr__(self, name, value)
        if name in vector_sizes and hasattr(self, "_slice_cache"):
            self._slice_cache.clear()

    def _coerce_covariance(
        self,
        value: Covariance | Any,
        *,
        name: str,
        default_coordinates: str,
    ) -> Covariance:
        if isinstance(value, Covariance):
            result = value.copy(
                coordinates=value.coordinates or default_coordinates,
                psd_policy="allow_indefinite",
            )
        else:
            matrix = _square_matrix(value, name=name)
            if matrix is None:
                raise ValueError(f"{name} cannot be None")
            result = Covariance(
                matrix,
                coordinates=default_coordinates,
                psd_policy="allow_indefinite",
            )
        allowed = self._allowed_covariance_shapes()
        if allowed is not None and result.shape not in allowed:
            raise ValueError(
                f"{name} must use reduced- or full-quaternion coordinates; "
                f"expected one of {sorted(allowed)}, got {result.shape}"
            )
        return result

    @property
    def covariance(self) -> Covariance:
        """State-estimation covariance in full or square-root form."""
        return self._covariance

    @covariance.setter
    def covariance(self, value: Covariance) -> None:
        replacement = self._coerce_covariance(
            value, name="cov", default_coordinates="state_tangent"
        )
        try:
            process_shape = self._process_noise.shape
        except AttributeError:
            process_shape = replacement.shape
        if replacement.shape != process_shape:
            raise ValueError(
                f"cov must match int_cov shape {process_shape}, got {replacement.shape}"
            )
        object.__setattr__(self, "_covariance", replacement)

    @property
    def process_noise(self) -> Covariance:
        """Process-noise covariance associated with this estimated state."""
        return self._process_noise

    @process_noise.setter
    def process_noise(self, value: Covariance) -> None:
        replacement = self._coerce_covariance(
            value, name="int_cov", default_coordinates=self._covariance.coordinates
        )
        if replacement.shape != self._covariance.shape:
            raise ValueError(
                f"int_cov must match cov shape {self._covariance.shape}, "
                f"got {replacement.shape}"
            )
        object.__setattr__(self, "_process_noise", replacement)

    @property
    def cov(self) -> np.ndarray:
        """Legacy full-matrix view of :attr:`covariance`."""
        return self._covariance.as_matrix()

    @cov.setter
    def cov(self, value: Any) -> None:
        matrix = _square_matrix(value, name="cov")
        if matrix is None:
            raise ValueError("cov cannot be None")
        self._validate_covariance_assignment("cov", matrix)
        self._covariance.assign(matrix)

    @property
    def int_cov(self) -> np.ndarray:
        """Legacy full-matrix view of :attr:`process_noise`."""
        return self._process_noise.as_matrix()

    @int_cov.setter
    def int_cov(self, value: Any) -> None:
        matrix = _square_matrix(value, name="int_cov")
        if matrix is None:
            raise ValueError("int_cov cannot be None")
        self._validate_covariance_assignment("int_cov", matrix)
        self._process_noise.assign(matrix)

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
        """Compatibility name for the stored estimator-state dimension."""
        return self.full_size

    @property
    def uses_reduced_quaternion_covariance(self) -> bool:
        return self.cov.shape == (self.tangent_size, self.tangent_size)

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
            covariance=self.covariance,
            process_noise=self.process_noise,
        )

    def normalized(self) -> EstimatorState:
        norm = float(np.linalg.norm(self.q))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("q must have a finite, non-zero norm to normalize")
        result = self.copy()
        result.q = result.q / norm
        return result

    def interpolate(self, other: State, alpha: float, *, method: str = "slerp") -> EstimatorState:
        """Blend two estimated states; see :meth:`~ADCS.state.State.interpolate`.

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
            covariance=Covariance(
                lerp(self.cov, other.cov),
                form=self.covariance.form,
                coordinates=self.covariance.coordinates,
                psd_policy="allow_indefinite",
            ),
            process_noise=Covariance(
                lerp(self.int_cov, other.int_cov),
                form=self.process_noise.form,
                coordinates=self.process_noise.coordinates,
                psd_policy="allow_indefinite",
            ),
        )

    def minus(
        self,
        ref: State,
        *,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
        quaternion_order: str = State.DEFAULT_QUATERNION_ORDER,
        shortest: bool = True,
    ) -> np.ndarray:
        """Return the augmented-state difference in reduced local coordinates."""
        if not isinstance(ref, EstimatorState):
            raise TypeError(f"ref must be an EstimatorState, got {type(ref).__name__}")
        for name in ("act_bias", "sens_bias", "dist_param"):
            if getattr(self, name).size != getattr(ref, name).size:
                raise ValueError(f"states must have matching {name} sizes to subtract")
        base = State.minus(
            self,
            ref,
            quaternion_mode=quaternion_mode,
            quaternion_order=quaternion_order,
            shortest=shortest,
        )
        return np.concatenate(
            (
                base,
                self.act_bias - ref.act_bias,
                self.sens_bias - ref.sens_bias,
                self.dist_param - ref.dist_param,
            )
        )

    def plus(
        self,
        delta: Any,
        *,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
        quaternion_order: str = State.DEFAULT_QUATERNION_ORDER,
        normalize: bool = True,
    ) -> EstimatorState:
        """Apply a perturbation to physical and estimated parameter blocks."""
        quaternion_mode = _quaternion_mode(quaternion_mode)
        if quaternion_mode == "full_quaternion":
            delta = _vector(delta, name="delta", size=self.full_size)
            result = self.copy()
            slices = self.slices(coordinates="full")
            result.w = self.w + delta[slices["angular_velocity"]]
            q = self.q + delta[slices["attitude"]]
            result.q = _unit_quaternion(q) if normalize else q
            for name, attribute in self._BLOCK_FIELDS[2:]:
                setattr(result, attribute, self.block(name) + delta[slices[name]])
            return result
        delta = _vector(delta, name="delta", size=self.tangent_size)
        slices = self.slices(coordinates="tangent")
        dq = _quat_delta_from_vector(delta[slices["attitude"]], quaternion_mode)
        result = self.with_quaternion_delta(
            dq,
            order=quaternion_order,
            normalize=normalize,
        )
        result.w = self.w + delta[slices["angular_velocity"]]
        for name, attribute in self._BLOCK_FIELDS[2:]:
            setattr(result, attribute, self.block(name) + delta[slices[name]])
        return result

    def subtract(self, ref: State) -> np.ndarray:
        """Compatibility wrapper for :meth:`~ADCS.state.EstimatorState.minus`."""
        return self.minus(ref)

    def add_error(self, delta: np.ndarray) -> EstimatorState:
        """Compatibility wrapper for :meth:`~ADCS.state.EstimatorState.plus`."""
        return self.plus(delta)

    def is_close(
        self,
        other: State,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        compare_covariance: bool = False,
    ) -> bool:
        """Compare augmented states, optionally including covariance matrices."""
        if not isinstance(other, EstimatorState) or not State.is_close(
            self, other, rtol=rtol, atol=atol
        ):
            return False
        blocks_close = all(
            getattr(self, name).shape == getattr(other, name).shape
            and np.allclose(getattr(self, name), getattr(other, name), rtol=rtol, atol=atol)
            for name in ("act_bias", "sens_bias", "dist_param")
        )
        if not blocks_close or not compare_covariance:
            return blocks_close
        return bool(
            self.cov.shape == other.cov.shape
            and self.int_cov.shape == other.int_cov.shape
            and np.allclose(self.cov, other.cov, rtol=rtol, atol=atol)
            and np.allclose(self.int_cov, other.int_cov, rtol=rtol, atol=atol)
        )

    def covariance_to_full(self, covariance: Any = None, **tangent_kwargs: Any) -> np.ndarray:
        r"""Project reduced covariance with :math:`P_f=G P_r G^T`.

        Here :math:`G` is :meth:`~ADCS.state.State.tangent_map`.
        """
        mode = tangent_kwargs.get("quaternion_mode", self.DEFAULT_QUATERNION_MODE)
        if _quaternion_mode(mode) == "full_quaternion":
            raise ValueError("covariance_to_full requires a reduced quaternion mode")
        source = self.cov if covariance is None else np.asarray(covariance, dtype=float)
        expected = (self.tangent_size, self.tangent_size)
        if source.shape != expected:
            raise ValueError(f"reduced covariance must have shape {expected}, got {source.shape}")
        tangent = self.tangent_map(**tangent_kwargs)
        result = tangent @ source @ tangent.T
        return (result + result.T) / 2.0

    def covariance_to_reduced(self, covariance: Any = None, **tangent_kwargs: Any) -> np.ndarray:
        r"""Project full covariance with :math:`P_r=G^\dagger P_f(G^\dagger)^T`.

        Here :math:`G^\dagger` is :meth:`~ADCS.state.State.tangent_pinv`.
        """
        mode = tangent_kwargs.get("quaternion_mode", self.DEFAULT_QUATERNION_MODE)
        if _quaternion_mode(mode) == "full_quaternion":
            raise ValueError("covariance_to_reduced requires a reduced quaternion mode")
        source = self.cov if covariance is None else np.asarray(covariance, dtype=float)
        expected = (self.full_size, self.full_size)
        if source.shape != expected:
            raise ValueError(f"full covariance must have shape {expected}, got {source.shape}")
        tangent_pinv = self.tangent_pinv(**tangent_kwargs)
        result = tangent_pinv @ source @ tangent_pinv.T
        return (result + result.T) / 2.0

    @classmethod
    def mean(
        cls,
        states: Iterable[EstimatorState],
        weights: Any = None,
        *,
        reference: EstimatorState | None = None,
        covariance: Literal["reference", "weighted"] = "reference",
        **kwargs: Any,
    ) -> EstimatorState:
        """Compute an augmented-state mean with an explicit covariance policy."""
        values = list(states)
        result = State.mean.__func__(cls, values, weights, reference=reference, **kwargs)
        if covariance == "reference":
            return result
        if covariance != "weighted":
            raise ValueError("covariance must be 'reference' or 'weighted'")
        if any(value.cov.shape != values[0].cov.shape for value in values):
            raise ValueError("states must use matching covariance shapes")
        if weights is None:
            weight_array = np.full(len(values), 1.0 / len(values))
        else:
            weight_array = _vector(weights, name="weights", size=len(values))
            weight_array = weight_array / np.sum(weight_array)
        result.cov = sum(
            (weight * value.cov for value, weight in zip(values, weight_array)),
            np.zeros_like(values[0].cov),
        )
        result.int_cov = sum(
            (weight * value.int_cov for value, weight in zip(values, weight_array)),
            np.zeros_like(values[0].int_cov),
        )
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
