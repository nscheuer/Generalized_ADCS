__all__ = ["EstimatedArray", "EstimatedOrbital_State"]

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from ADCS.orbits.orbital_state import Orbital_State

@dataclass
class EstimatedArray:
    """
    Represents an estimated vector with associated covariance and integrated covariance.

    Attributes
    ----------
    val : np.ndarray
        The estimated state or parameter vector.
    cov : np.ndarray
        The covariance matrix representing uncertainty in `val`.
    int_cov : np.ndarray
        The integrated process covariance (e.g., accumulated process noise).
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
        """Extract a sub-estimate and associated covariance blocks."""
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
        """Insert values and covariance blocks back into this estimate."""
        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        self.val[inds_mask] = val
        self.cov[np.ix_(cov_inds_mask, cov_inds_mask)] = cov
        self.int_cov[np.ix_(cov_inds_mask, cov_inds_mask)] = int_cov

    def copy(self):
        """Return a deep copy."""
        return EstimatedArray(
            self.val.copy(),
            self.cov.copy(),
            self.int_cov.copy(),
        )
    

@dataclass
class EstimatedOrbital_State:
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
            raise ValueError(f"P must be 6×6, got {self.P.shape}")
        if self.Q.shape != (6, 6):
            raise ValueError(f"Q must be 6×6, got {self.Q.shape}")
        

    def pull_indices(self, inds_mask, cov_missing_inds=None):
        inds_mask = np.asarray(inds_mask)

        if cov_missing_inds is None:
            cov_inds_mask = inds_mask
        else:
            cov_inds_mask = np.delete(inds_mask, cov_missing_inds)

        # pull out R and V based on mask (first 3 → R, last 3 → V)
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
        """
        Insert values and covariance blocks back into the full estimate.
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
        return EstimatedOrbital_State(
            self.os.copy(),
            self.P.copy(),
            self.Q.copy(),
        )