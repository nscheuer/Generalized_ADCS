__all__ = ["EstimatedArray"]

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

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