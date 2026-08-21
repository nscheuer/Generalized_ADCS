r"""Covariance storage and common estimator covariance operations.

The public :class:`Covariance` interface is independent of whether uncertainty
is stored as a full matrix or as an upper-triangular square-root factor.  This
keeps representation details out of estimators and hardware models.
"""

from __future__ import annotations

__all__ = ["Covariance"]

from typing import Any, Iterable, Literal

import numpy as np
from scipy.linalg import solve_triangular

from ADCS.helpers.cholesky_update import choldowndate, cholupdate


CovarianceForm = Literal["full", "sqrt"]
PSDPolicy = Literal["strict", "project", "jitter"]


def _form(value: str) -> CovarianceForm:
    if value not in ("full", "sqrt"):
        raise ValueError(f"form must be 'full' or 'sqrt', got {value!r}")
    return value  # type: ignore[return-value]


def _policy(value: str) -> PSDPolicy:
    if value not in ("strict", "project", "jitter"):
        raise ValueError(
            f"psd_policy must be 'strict', 'project', or 'jitter', got {value!r}"
        )
    return value  # type: ignore[return-value]


def _square(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _symmetric_psd(matrix: Any, *, policy: PSDPolicy) -> np.ndarray:
    matrix = _square(matrix, name="covariance")
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2))) if matrix.size else 1.0
    tolerance = 1e-12 * scale
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=tolerance):
        raise ValueError("covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2.0
    if not matrix.size:
        return matrix

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum = float(eigenvalues[0])
    if policy == "strict" and minimum < -tolerance:
        raise ValueError(f"covariance must be positive semidefinite; minimum eigenvalue={minimum}")
    if policy == "project":
        eigenvalues = np.maximum(eigenvalues, 0.0)
        matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
    elif policy == "jitter" and minimum <= tolerance:
        matrix = matrix + np.eye(matrix.shape[0]) * (tolerance - minimum + tolerance)
    return (matrix + matrix.T) / 2.0


def _normalize_factor(factor: np.ndarray) -> np.ndarray:
    factor = np.triu(np.array(factor, dtype=float, copy=True))
    for i in range(factor.shape[0]):
        if factor[i, i] < 0.0:
            factor[i] *= -1.0
    return factor


def _safe_upper_cholesky(matrix: np.ndarray, *, policy: PSDPolicy) -> np.ndarray:
    matrix = _symmetric_psd(matrix, policy=policy)
    if not matrix.size:
        return matrix
    try:
        return np.linalg.cholesky(matrix).T
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        tolerance = 1e-12 * scale
        if policy == "strict" and float(eigenvalues[0]) < -tolerance:
            raise ValueError("covariance must be positive semidefinite") from None
        eigenvalues = np.maximum(eigenvalues, 0.0)
        root = eigenvectors * np.sqrt(eigenvalues)
        _, upper = np.linalg.qr(root.T)
        return _normalize_factor(upper)


def _as_covariance_matrix(value: Covariance | Any, *, name: str) -> np.ndarray:
    if isinstance(value, Covariance):
        return value.as_matrix()
    return _square(value, name=name)


class Covariance:
    r"""Own a covariance in full or upper square-root form.

    For ``form="sqrt"``, the internal factor :math:`S` is upper triangular and
    satisfies :math:`P=S^T S`. Inputs and returned matrices are copied, so
    updates occur only through explicit methods such as :meth:`assign`.

    :param matrix: Symmetric positive-semidefinite covariance matrix.
    :param form: Internal representation, ``"full"`` or ``"sqrt"``.
    :param coordinates: Descriptive coordinate-space label.
    :param psd_policy: Handling for numerically indefinite matrices.
    """

    __slots__ = ("_data", "_form", "_coordinates", "_psd_policy")

    def __init__(
        self,
        matrix: Any,
        *,
        form: str = "full",
        coordinates: str = "generic",
        psd_policy: str = "strict",
    ) -> None:
        self._form = _form(form)
        self._coordinates = str(coordinates)
        self._psd_policy = _policy(psd_policy)
        validated = _symmetric_psd(matrix, policy=self._psd_policy)
        self._data = (
            validated
            if self._form == "full"
            else _safe_upper_cholesky(validated, policy=self._psd_policy)
        )

    @classmethod
    def from_matrix(cls, matrix: Any, **kwargs: Any) -> Covariance:
        """Construct from a covariance matrix."""
        return cls(matrix, **kwargs)

    @classmethod
    def from_upper_factor(
        cls,
        factor: Any,
        *,
        form: str = "sqrt",
        coordinates: str = "generic",
        psd_policy: str = "strict",
    ) -> Covariance:
        r"""Construct from upper :math:`S` satisfying :math:`P=S^T S`."""
        factor = _square(factor, name="upper factor")
        if not np.allclose(factor, np.triu(factor), rtol=0.0, atol=1e-12):
            raise ValueError("upper factor must be upper triangular")
        result = cls.__new__(cls)
        result._form = _form(form)
        result._coordinates = str(coordinates)
        result._psd_policy = _policy(psd_policy)
        factor = _normalize_factor(factor)
        result._data = factor if result._form == "sqrt" else factor.T @ factor
        return result

    @classmethod
    def zeros(cls, dimension: int, **kwargs: Any) -> Covariance:
        if dimension < 0:
            raise ValueError("dimension cannot be negative")
        return cls(np.zeros((dimension, dimension)), **kwargs)

    @classmethod
    def identity(cls, dimension: int, scale: float = 1.0, **kwargs: Any) -> Covariance:
        if dimension < 0 or scale < 0.0:
            raise ValueError("dimension and scale must be non-negative")
        return cls(np.eye(dimension) * scale, **kwargs)

    @classmethod
    def block_diagonal(
        cls,
        blocks: Iterable[Covariance | Any],
        **kwargs: Any,
    ) -> Covariance:
        matrices = [_as_covariance_matrix(block, name="covariance block") for block in blocks]
        size = sum(matrix.shape[0] for matrix in matrices)
        result = np.zeros((size, size))
        offset = 0
        for matrix in matrices:
            width = matrix.shape[0]
            result[offset : offset + width, offset : offset + width] = matrix
            offset += width
        return cls(result, **kwargs)

    @classmethod
    def from_weighted_deviations(
        cls,
        deviations: Any,
        weights: Any,
        noise: Covariance | Any | None = None,
        **kwargs: Any,
    ) -> Covariance:
        deviations, weights = cls._deviations_and_weights(deviations, weights)
        dimension = deviations.shape[1]
        matrix = np.einsum("i,ij,ik->jk", weights, deviations, deviations)
        if noise is not None:
            noise_matrix = _as_covariance_matrix(noise, name="noise covariance")
            if noise_matrix.shape != (dimension, dimension):
                raise ValueError("noise covariance dimension must match deviations")
            matrix += noise_matrix
        return cls((matrix + matrix.T) / 2.0, **kwargs)

    @staticmethod
    def _deviations_and_weights(deviations: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
        deviations = np.asarray(deviations, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if deviations.ndim != 2:
            raise ValueError("deviations must have shape (samples, dimension)")
        if weights.ndim != 1 or weights.shape[0] != deviations.shape[0]:
            raise ValueError("weights must contain one value per deviation")
        if not np.all(np.isfinite(deviations)) or not np.all(np.isfinite(weights)):
            raise ValueError("deviations and weights must contain only finite values")
        return deviations, weights

    @property
    def form(self) -> CovarianceForm:
        return self._form

    @property
    def coordinates(self) -> str:
        return self._coordinates

    @property
    def dimension(self) -> int:
        return self._data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def __len__(self) -> int:
        return self.dimension

    def __repr__(self) -> str:
        return (
            f"Covariance(dimension={self.dimension}, form={self.form!r}, "
            f"coordinates={self.coordinates!r})"
        )

    def copy(self, *, form: str | None = None, coordinates: str | None = None) -> Covariance:
        target_form = self.form if form is None else _form(form)
        target_coordinates = self.coordinates if coordinates is None else coordinates
        if target_form == "sqrt":
            return Covariance.from_upper_factor(
                self.upper_factor(),
                form="sqrt",
                coordinates=target_coordinates,
                psd_policy=self._psd_policy,
            )
        return Covariance(
            self.as_matrix(),
            form="full",
            coordinates=target_coordinates,
            psd_policy=self._psd_policy,
        )

    def as_matrix(self) -> np.ndarray:
        """Return an owned full covariance matrix."""
        if self.form == "full":
            return self._data.copy()
        return self._data.T @ self._data

    def upper_factor(self) -> np.ndarray:
        r"""Return an owned upper factor :math:`S`, where :math:`P=S^T S`."""
        if self.form == "sqrt":
            return self._data.copy()
        return _safe_upper_cholesky(self._data, policy=self._psd_policy)

    def assign(self, matrix: Covariance | Any) -> None:
        """Atomically replace the covariance while retaining this object's form."""
        matrix = _as_covariance_matrix(matrix, name="covariance")
        validated = _symmetric_psd(matrix, policy=self._psd_policy)
        replacement = (
            validated
            if self.form == "full"
            else _safe_upper_cholesky(validated, policy=self._psd_policy)
        )
        self._data = replacement

    def assign_upper_factor(self, factor: Any) -> None:
        """Atomically replace the covariance from an upper factor."""
        replacement = Covariance.from_upper_factor(
            factor,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )
        self._data = replacement._data

    def sigma_offsets(self, scale: float = 1.0) -> np.ndarray:
        """Return paired positive and negative sigma-point offsets as rows."""
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("scale must be finite and non-negative")
        positive = scale * self.upper_factor()
        return np.vstack((positive, -positive))

    def solve(self, rhs: Any) -> np.ndarray:
        r"""Solve :math:`P X=B` without exposing the stored representation."""
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.dimension:
            raise ValueError("rhs leading dimension must match covariance dimension")
        if self.form == "full":
            return np.linalg.solve(self._data, rhs)
        intermediate = solve_triangular(self._data.T, rhs, lower=True)
        return solve_triangular(self._data, intermediate, lower=False)

    def transformed(self, jacobian: Any, *, coordinates: str | None = None) -> Covariance:
        r"""Return :math:`J P J^T`."""
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[1] != self.dimension:
            raise ValueError("jacobian column count must match covariance dimension")
        result = jacobian @ self.as_matrix() @ jacobian.T
        return Covariance(
            result,
            form=self.form,
            coordinates=self.coordinates if coordinates is None else coordinates,
            psd_policy=self._psd_policy,
        )

    def subset(self, indices: Any, *, coordinates: str | None = None) -> Covariance:
        indices = np.arange(self.dimension)[indices]
        indices = np.atleast_1d(indices).astype(int)
        matrix = self.as_matrix()[np.ix_(indices, indices)]
        return Covariance(
            matrix,
            form=self.form,
            coordinates=self.coordinates if coordinates is None else coordinates,
            psd_policy=self._psd_policy,
        )

    def replace_block(self, indices: Any, block: Covariance | Any) -> None:
        selected = np.atleast_1d(np.arange(self.dimension)[indices]).astype(int)
        block_matrix = _as_covariance_matrix(block, name="covariance block")
        if block_matrix.shape != (selected.size, selected.size):
            raise ValueError("replacement block shape must match selected indices")
        matrix = self.as_matrix()
        matrix[np.ix_(selected, selected)] = block_matrix
        self.assign(matrix)

    def zero_cross(self, first: Any, second: Any) -> None:
        """Set covariance cross terms between two index selections to zero."""
        first = np.atleast_1d(np.arange(self.dimension)[first]).astype(int)
        second = np.atleast_1d(np.arange(self.dimension)[second]).astype(int)
        matrix = self.as_matrix()
        matrix[np.ix_(first, second)] = 0.0
        matrix[np.ix_(second, first)] = 0.0
        self.assign(matrix)

    def added(self, other: Covariance | Any) -> Covariance:
        other_matrix = _as_covariance_matrix(other, name="covariance")
        if other_matrix.shape != self.shape:
            raise ValueError("covariances must have matching shapes")
        return Covariance(
            self.as_matrix() + other_matrix,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    def scaled(self, scale: float) -> Covariance:
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("covariance scale must be finite and non-negative")
        return Covariance(
            scale * self.as_matrix(),
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    def rank_updated(self, vectors: Any, weight: float = 1.0) -> Covariance:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (updates, dimension)")
        weight = float(weight)
        matrix = self.as_matrix() + weight * vectors.T @ vectors
        return Covariance(
            matrix,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    def predicted_linear(self, transition: Any, noise: Covariance | Any) -> Covariance:
        r"""Return the linear prediction :math:`F P F^T+Q`."""
        transition = np.asarray(transition, dtype=float)
        if transition.ndim != 2 or transition.shape[1] != self.dimension:
            raise ValueError("transition column count must match covariance dimension")
        noise_matrix = _as_covariance_matrix(noise, name="process noise covariance")
        if noise_matrix.shape != (transition.shape[0], transition.shape[0]):
            raise ValueError("process noise dimension must match transition output")
        matrix = transition @ self.as_matrix() @ transition.T + noise_matrix
        return Covariance(
            matrix,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    def predicted_unscented(
        self,
        deviations: Any,
        weights: Any,
        noise: Covariance | Any,
    ) -> Covariance:
        """Return a weighted sigma-deviation covariance plus process noise."""
        return Covariance.from_weighted_deviations(
            deviations,
            weights,
            noise,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    def updated_linear(
        self,
        measurement_jacobian: Any,
        measurement_noise: Covariance | Any,
        *,
        joseph: bool = True,
    ) -> tuple[np.ndarray, Covariance]:
        r"""Return Kalman gain and posterior covariance for a linear update."""
        h = np.asarray(measurement_jacobian, dtype=float)
        if h.ndim != 2 or h.shape[1] != self.dimension:
            raise ValueError("measurement jacobian column count must match state dimension")
        r = _as_covariance_matrix(measurement_noise, name="measurement noise covariance")
        if r.shape != (h.shape[0], h.shape[0]):
            raise ValueError("measurement noise dimension must match measurement jacobian")
        p = self.as_matrix()
        innovation = Covariance(h @ p @ h.T + r, psd_policy=self._psd_policy)
        gain = innovation.solve(h @ p).T
        identity = np.eye(self.dimension)
        if joseph:
            residual = identity - gain @ h
            posterior = residual @ p @ residual.T + gain @ r @ gain.T
        else:
            posterior = (identity - gain @ h) @ p
        posterior = (posterior + posterior.T) / 2.0
        return gain, Covariance(
            posterior,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )

    @staticmethod
    def cross_covariance(
        first_deviations: Any,
        second_deviations: Any,
        weights: Any,
    ) -> np.ndarray:
        first, weights = Covariance._deviations_and_weights(first_deviations, weights)
        second = np.asarray(second_deviations, dtype=float)
        if second.ndim != 2 or second.shape[0] != first.shape[0]:
            raise ValueError("deviation arrays must contain the same number of samples")
        if not np.all(np.isfinite(second)):
            raise ValueError("deviations must contain only finite values")
        return np.einsum("i,ij,ik->jk", weights, first, second)

    def updated_unscented(
        self,
        state_deviations: Any,
        measurement_deviations: Any,
        weights: Any,
        measurement_noise: Covariance | Any,
    ) -> tuple[np.ndarray, Covariance]:
        """Return Kalman gain and posterior from weighted sigma deviations."""
        state_deviations, weights = self._deviations_and_weights(state_deviations, weights)
        if state_deviations.shape[1] != self.dimension:
            raise ValueError("state deviation dimension must match covariance")
        measurement_deviations = np.asarray(measurement_deviations, dtype=float)
        cross = self.cross_covariance(state_deviations, measurement_deviations, weights)
        innovation = Covariance.from_weighted_deviations(
            measurement_deviations,
            weights,
            measurement_noise,
            form=self.form,
            coordinates="measurement",
            psd_policy=self._psd_policy,
        )
        gain = innovation.solve(cross.T).T
        posterior = self.as_matrix() - gain @ innovation.as_matrix() @ gain.T
        posterior = (posterior + posterior.T) / 2.0
        return gain, Covariance(
            posterior,
            form=self.form,
            coordinates=self.coordinates,
            psd_policy=self._psd_policy,
        )
