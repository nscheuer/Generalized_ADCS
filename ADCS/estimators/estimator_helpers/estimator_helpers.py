__all__ = ["EstimatedOrbital_State"]

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from ADCS.orbits.orbital_state import Orbital_State

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
