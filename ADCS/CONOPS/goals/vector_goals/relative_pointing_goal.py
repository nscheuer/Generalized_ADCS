__all__ = ["Relative_Pointing_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize


class Relative_Pointing_Goal(Vector_Goal):
    r"""
    Point a boresight at another satellite in the formation (formation-aware goal).

    The desired inertial direction is the line of sight from this satellite to a
    target satellite, read from a shared
    :class:`~ADCS.formation.formation_world.FormationWorld` at evaluation time:

    .. math::

        \hat{\mathbf{r}}_{\mathrm{LOS}}(t)
        = \frac{\mathbf{r}_{\mathrm{target}}(t) - \mathbf{r}_{\mathrm{self}}(t)}
               {\lVert \mathbf{r}_{\mathrm{target}}(t) - \mathbf{r}_{\mathrm{self}}(t)\rVert}.

    A feed-forward reference angular velocity tracks the rotation of the
    line-of-sight vector (mirroring :class:`~ADCS.CONOPS.goals.Nadir_Goal`):

    .. math::

        \boldsymbol{\omega}_{\mathrm{ref}}
        = \frac{\mathbf{r}_{\mathrm{LOS}} \times \dot{\mathbf{r}}_{\mathrm{LOS}}}
               {\lVert \mathbf{r}_{\mathrm{LOS}}\rVert^2},
        \qquad
        \dot{\mathbf{r}}_{\mathrm{LOS}} = \mathbf{v}_{\mathrm{target}} - \mathbf{v}_{\mathrm{self}}.

    The goal closes over the shared world rather than changing the ``to_ref``
    signature: the orchestrator refreshes the world each timestep before goals
    are evaluated, so ``world.position(target_id)`` is the neighbour's current
    truth state.

    :param world: Shared :class:`~ADCS.formation.formation_world.FormationWorld`.
    :param target_id: Identifier of the satellite to point at.
    :param boresight_name: Optional boresight name (see :class:`Vector_Goal`).
    :param use_estimate: Read the target's estimated state instead of truth.
    """

    def __init__(self, world, target_id, boresight_name: str | None = None, use_estimate: bool = False) -> None:
        super().__init__(boresight_name=boresight_name)
        self.world = world
        self.target_id = target_id
        self.use_estimate = bool(use_estimate)

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r_target = self.world.position(self.target_id, estimate=self.use_estimate)
        v_target = self.world.velocity(self.target_id, estimate=self.use_estimate)

        los = np.asarray(r_target, dtype=float) - np.asarray(os0.R, dtype=float)
        rel_v = np.asarray(v_target, dtype=float) - np.asarray(os0.V, dtype=float)

        los2 = float(np.dot(los, los))
        if los2 <= 0.0:
            # Degenerate (co-located); fall back to current radial direction.
            r_goal = normalize(os0.R)
            w_ref = np.zeros(3)
        else:
            r_goal = los / np.sqrt(los2)
            w_ref = np.cross(los, rel_v) / los2

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = r_goal
        return r_ref, w_ref
