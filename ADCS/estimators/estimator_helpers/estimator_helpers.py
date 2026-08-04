__all__ = ["EstimatedArray", "EstimatedOrbital_State"]

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from ADCS.orbits.orbital_state import Orbital_State

@dataclass
class EstimatedArray:
    r"""
    Represents an estimated vector with associated covariance and integrated covariance.

    This structure is commonly used for parameter estimation where the state is a generic vector
    rather than a specific orbital state.

    :param val: The estimated state or parameter vector :math:`\hat{\mathbf{x}}`.
    :type val: np.ndarray
    :param cov: The covariance matrix :math:`P` representing uncertainty in ``val``.
                Must be square and match the dimensions of ``val``.

                .. warning::
                   For a reduced attitude estimator (non-``quat_as_vec``) the
                   attitude-error block ``cov[3:6, 3:6]`` is expressed in the
                   estimator's ``vec_mode`` 3-parameter, **not** the physical
                   attitude angle. It differs from an angle (rad\\ :sup:`2`)
                   covariance by :math:`G^{-2}` (= 4 for ``vec_mode=6``).
                   Do **not** read it (or ``sqrt`` of it) as an attitude angle
                   uncertainty. Use the sanctioned converters
                   :func:`~ADCS.helpers.math_helpers.attitude_angle_covariance`
                   / ``attitude_angle_sigma`` /
                   ``sample_attitude_error_rotvec``.
    :type cov: np.ndarray
    :param int_cov: The integrated process covariance :math:`Q_{int}` (e.g., accumulated process noise over time).
    :type int_cov: np.ndarray
    """
    val: np.ndarray
    cov: np.ndarray = field(default_factory=lambda: None)
    int_cov: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        self.val = np.asarray(self.val, dtype=float)
        n = self.val.size

        if self.cov is None:
            # default: same dimension as state
            self.cov = np.zeros((n, n))
        else:
            self.cov = np.asarray(self.cov, dtype=float)

        if self.int_cov is None:
            self.int_cov = np.zeros_like(self.cov)
        else:
            self.int_cov = np.asarray(self.int_cov, dtype=float)

        # Sanity checks, but no longer force (n,n):
        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError(f"cov must be square, got {self.cov.shape}")
        if self.int_cov.shape != self.cov.shape:
            raise ValueError(
                f"int_cov must have same shape as cov, got {self.int_cov.shape} vs {self.cov.shape}"
            )

    # ---- Methods ----

    def pull_indices(self, inds_mask, cov_missing_inds=None):
        """
        Extracts a sub-estimate and associated covariance blocks based on a mask.

        This is useful for isolating specific parameters from a larger state vector.

        :param inds_mask: Indices of the values to extract.
        :type inds_mask: list or np.ndarray
        :param cov_missing_inds: Indices to exclude from the covariance extraction (optional).
                                 If None, ``inds_mask`` is used for covariance as well.
        :type cov_missing_inds: list or np.ndarray, optional
        :return: A new EstimatedArray containing only the selected elements.
        :rtype: ~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedArray
        """
        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        return EstimatedArray(
            self.val[inds_mask],
            self.cov[np.ix_(cov_inds_mask, cov_inds_mask)],
            self.int_cov[np.ix_(cov_inds_mask, cov_inds_mask)],
        )

    def set_indices(self, inds_mask, val, cov, int_cov, cov_missing_inds=None):
        """
        Inserts values and covariance blocks back into this estimate.

        :param inds_mask: Indices in the full vector where data should be inserted.
        :type inds_mask: list or np.ndarray
        :param val: The sub-vector of values to insert.
        :type val: np.ndarray
        :param cov: The covariance block corresponding to the inserted values.
        :type cov: np.ndarray
        :param int_cov: The integrated covariance block to insert.
        :type int_cov: np.ndarray
        :param cov_missing_inds: Indices to exclude from the covariance insertion mapping (optional).
        :type cov_missing_inds: list or np.ndarray, optional
        """
        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        self.val[inds_mask] = val
        self.cov[np.ix_(cov_inds_mask, cov_inds_mask)] = cov
        self.int_cov[np.ix_(cov_inds_mask, cov_inds_mask)] = int_cov

    def copy(self):
        """
        Creates a deep copy of the EstimatedArray.

        :return: A new instance with independent data copies.
        :rtype: ~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedArray
        """
        return EstimatedArray(
            self.val.copy(),
            self.cov.copy(),
            self.int_cov.copy(),
        )
    

@dataclass
class EstimatedOrbital_State:
    r"""
    A container for an Orbital State and its associated estimation uncertainty.

    This class wraps the physical :class:`~ADCS.orbits.orbital_state.Orbital_State` with
    estimation statistics, specifically the state covariance :math:`P` and process noise :math:`Q`.

    :param os: The estimated physical state (Position :math:`\mathbf{r}` and Velocity :math:`\mathbf{v}`).
    :type os: ~ADCS.orbits.orbital_state.Orbital_State
    :param P: The :math:`6 \times 6` state covariance matrix.
              .. math:: P = E[(\hat{x} - x)(\hat{x} - x)^T]
    :type P: np.ndarray
    :param Q: The :math:`6 \times 6` process noise covariance matrix.
    :type Q: np.ndarray
    """
    os: Orbital_State
    P: np.ndarray = field(default_factory=lambda: None)
    Q: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        if self.P is None:
            self.P = np.zeros((6, 6))
        else:
            self.P = np.asarray(self.P, dtype=float)

        if self.Q is None:
            self.Q = np.zeros_like(self.P)
        else:
            self.Q = np.asarray(self.Q, dtype=float)
        if self.P.shape != (6, 6):
            raise ValueError(f"P must be 6x6, got {self.P.shape}")
        if self.Q.shape != (6, 6):
            raise ValueError(f"Q must be 6x6, got {self.Q.shape}")
        

    def pull_indices(self, inds_mask, cov_missing_inds=None):
        """
        Extracts a partial state estimate based on indices mapping to [R, V].

        The mapping is linear: indices 0-2 correspond to Position (R), and 3-5 to Velocity (V).

        :param inds_mask: The indices of the 6-element state vector to extract.
        :type inds_mask: list or np.ndarray
        :param cov_missing_inds: Indices to exclude from covariance slicing (optional).
        :type cov_missing_inds: list or np.ndarray, optional
        :return: A new EstimatedOrbital_State containing the subset of state and covariance.
                 Note: The underlying Orbital_State will have 0.0 in unselected components.
        :rtype: ~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedOrbital_State
        """
        inds_mask = np.asarray(inds_mask)

        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        # pull out R and V based on mask (first 3 -> R, last 3 -> V)
        new_R = self.os.R.copy()
        new_V = self.os.V.copy()

        # modify based on mask
        for i, idx in enumerate(inds_mask):
            if idx < 3:
                new_R[idx] = np.hstack([self.os.R, self.os.V])[inds_mask][i]
            else:
                new_V[idx - 3] = np.hstack([self.os.R, self.os.V])[inds_mask][i]

        # new Orbital_State
        new_os = self.os.copy()
        new_os.R = new_R
        new_os.V = new_V

        return EstimatedOrbital_State(
            os=new_os,
            P=self.P[np.ix_(cov_inds_mask, cov_inds_mask)],
            Q=self.Q[np.ix_(cov_inds_mask, cov_inds_mask)],
        )


    def set_indices(self, inds_mask, val, P, Q, cov_missing_inds=None):
        r"""
        Inserts values and covariance blocks back into the full 6-state estimate.

        :param inds_mask: Indices in the full 6-element vector where data should be inserted.
        :type inds_mask: list or np.ndarray
        :param val: The values to insert into Position/Velocity.
        :type val: np.ndarray
        :param P: The covariance block :math:`P_{sub}` to insert.
        :type P: np.ndarray
        :param Q: The process noise block :math:`Q_{sub}` to insert.
        :type Q: np.ndarray
        :param cov_missing_inds: Indices to exclude from covariance insertion (optional).
        :type cov_missing_inds: list or np.ndarray, optional
        """

        inds_mask = np.asarray(inds_mask)

        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        # insert R and V
        full_x = np.hstack([self.os.R, self.os.V])
        full_x[inds_mask] = val

        self.os.R = full_x[:3]
        self.os.V = full_x[3:]

        # update covariance blocks
        self.P[np.ix_(cov_inds_mask, cov_inds_mask)] = P
        self.Q[np.ix_(cov_inds_mask, cov_inds_mask)] = Q

    def copy(self):
        """
        Creates a deep copy of the EstimatedOrbital_State.

        :return: A new instance with deep-copied state and matrices.
        :rtype: ~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedOrbital_State
        """
        return EstimatedOrbital_State(
            self.os.copy(),
            self.P.copy(),
            self.Q.copy(),
        )