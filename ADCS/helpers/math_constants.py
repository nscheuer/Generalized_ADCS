from __future__ import annotations
__all__ = ["MathConstants"]

import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MathConstants:
    r"""
    Container for commonly used mathematical constants in ADCS computations.

    This immutable dataclass provides standard vectors and quaternions that are
    frequently reused throughout the attitude determination and control codebase.

    ======================
    Defined Constants
    ======================

    * ``unitvecs``:
      Canonical Cartesian unit vectors

      .. math::

          \mathbf{e}_1 = [1, 0, 0]^T,\quad
          \mathbf{e}_2 = [0, 1, 0]^T,\quad
          \mathbf{e}_3 = [0, 0, 1]^T

    * ``zeroquat``:
      Identity quaternion representing zero rotation

      .. math::

          \mathbf{q}_0 = [1, 0, 0, 0]^T

    :return:
        Immutable namespace of mathematical constants.
    :rtype:
        MathConstants

    """
    unitvecs = [np.eye(3)[:,j] for j in list(range(3))]
    zeroquat  = np.array([1.0,0.0,0.0,0.0])