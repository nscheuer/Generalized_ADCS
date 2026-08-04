__all__ = ["FormationWorld"]

import numpy as np
from typing import Dict, Optional


class FormationWorld:
    r"""
    Shared, per-timestep registry of every satellite's current state.

    The orchestrator (:class:`~ADCS.formation.constellation.Constellation`)
    refreshes this registry at the top of every timestep, *before* any
    satellite is stepped, so that formation-aware goals (e.g. "point at a
    neighbour") read a consistent synchronous snapshot of the constellation.

    Goals hold a reference to one ``FormationWorld`` instance and query it by
    satellite id inside their ``to_ref`` — no goal API change is required.
    Stored quantities are inertial (ECI) truth (and optionally estimates):
    position ``R`` [km], velocity ``V`` [km/s], and attitude quaternion ``q``.
    """

    def __init__(self) -> None:
        self._truth: Dict[object, Dict[str, np.ndarray]] = {}
        self._estimate: Dict[object, Dict[str, np.ndarray]] = {}
        self.time_J2000: Optional[float] = None

    def update(self, sat_id, R, V, q=None, *, estimate: bool = False, J2000: Optional[float] = None) -> None:
        r"""
        Publish one satellite's current state into the registry.

        :param sat_id: Satellite identifier (any hashable).
        :param R: ECI position [km], shape ``(3,)``.
        :param V: ECI velocity [km/s], shape ``(3,)``.
        :param q: Attitude quaternion, shape ``(4,)`` (optional).
        :param estimate: Store into the estimate registry rather than truth.
        :param J2000: Optional epoch stamp for the snapshot.
        """
        rec = {
            "R": np.asarray(R, dtype=float).reshape(3).copy(),
            "V": np.asarray(V, dtype=float).reshape(3).copy(),
        }
        if q is not None:
            rec["q"] = np.asarray(q, dtype=float).reshape(4).copy()
        (self._estimate if estimate else self._truth)[sat_id] = rec
        if J2000 is not None:
            self.time_J2000 = float(J2000)

    def _get(self, sat_id, key, estimate):
        store = self._estimate if estimate else self._truth
        if sat_id not in store:
            raise KeyError(f"FormationWorld has no {'estimate' if estimate else 'truth'} for satellite {sat_id!r}")
        return store[sat_id][key]

    def position(self, sat_id, estimate: bool = False) -> np.ndarray:
        r"""ECI position [km] of satellite ``sat_id``."""
        return self._get(sat_id, "R", estimate)

    def velocity(self, sat_id, estimate: bool = False) -> np.ndarray:
        r"""ECI velocity [km/s] of satellite ``sat_id``."""
        return self._get(sat_id, "V", estimate)

    def quaternion(self, sat_id, estimate: bool = False) -> np.ndarray:
        r"""Attitude quaternion of satellite ``sat_id``."""
        return self._get(sat_id, "q", estimate)

    def ids(self):
        r"""Iterable of known truth satellite ids."""
        return list(self._truth.keys())

    def has(self, sat_id, estimate: bool = False) -> bool:
        return sat_id in (self._estimate if estimate else self._truth)
