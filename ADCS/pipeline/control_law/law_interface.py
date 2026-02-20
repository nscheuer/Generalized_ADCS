"""
Control Law abstract interface.

A ControlLaw is a black-box mapping from attitude/omega error signals
to a desired torque (or actuator command). The LawInterface dataclass
declares what inputs the law expects and what it produces, allowing
the goal formulation and compensation blocks to adapt automatically.
"""

__all__ = ["ControlLaw", "LawInterface"]

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ADCS.pipeline.data import LawInterface


class ControlLaw(ABC):
    """Abstract base class for pipeline-compatible control laws.

    Subclasses implement a specific control law (PD, sliding mode, etc.)
    and declare their interface via the ``interface`` property.
    """

    @property
    @abstractmethod
    def interface(self) -> LawInterface:
        """Return the LawInterface declaring this law's expectations."""
        ...

    @abstractmethod
    def compute(
        self,
        attitude_input: np.ndarray,
        omega_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the control output.

        Parameters
        ----------
        attitude_input : ndarray
            Attitude error vector (3,) for full laws, or
            (b_hat, r_target) tuple for reduced laws.
        omega_input : ndarray or None
            Angular velocity error, raw omega, or None depending on
            ``interface.omega_type``.

        Returns
        -------
        ndarray, shape (3,)
            Desired torque in body frame (if output_type='torque'),
            or actuator commands (if output_type='actuator_commands').
        """
        ...
