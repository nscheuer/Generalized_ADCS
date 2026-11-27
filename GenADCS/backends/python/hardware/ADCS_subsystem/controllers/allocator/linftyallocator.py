"""L-infinity (min-max) reaction-wheel torque allocator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit

from .allocator import ALLOC_LINFINITY, Allocator, saturate_wheels


@njit(cache=True)
def allocate_linfinity(
    tau_c: np.ndarray,
    a_pinv: np.ndarray,
    null_basis: np.ndarray,
    null_proj: np.ndarray,
    u_0: np.ndarray,
    max_torque: np.ndarray,
) -> np.ndarray:
    """Fast L-infinity (min-max) torque allocation + nullspace desaturation."""
    u_base = a_pinv @ tau_c
    r = null_basis.shape[1]
    if r == 0:
        return saturate_wheels(u_base, max_torque)

    n_w = u_base.size
    if r == 1:
        v = null_basis[:, 0]
        best_u = u_base.copy()
        best_max = 0.0
        for i in range(n_w):
            a = abs(best_u[i])
            if a > best_max:
                best_max = a
        for i in range(n_w):
            for j in range(i + 1, n_w):
                for sgn in (1.0, -1.0):
                    denom = v[i] - sgn * v[j]
                    if abs(denom) > 1e-12:
                        z = -(u_base[i] - sgn * u_base[j]) / denom
                        cand_max = 0.0
                        for k in range(n_w):
                            a = abs(u_base[k] + z * v[k])
                            if a > cand_max:
                                cand_max = a
                        if cand_max < best_max:
                            best_max = cand_max
                            best_u = u_base + z * v
        u = best_u
    else:
        # General r >= 2: subgradient minimization of ||u_base + V z||_inf,
        # tracking best_u so result is guaranteed <= ||u_base||_inf.
        z = np.zeros(r, dtype=np.float64)
        u = u_base.copy()
        best_u = u_base.copy()
        best_max = 0.0
        for i in range(n_w):
            a = abs(best_u[i])
            if a > best_max:
                best_max = a
        for it in range(40):
            k_max = 0
            u_max = abs(u[0])
            for i in range(1, n_w):
                a = abs(u[i])
                if a > u_max:
                    u_max = a
                    k_max = i
            if u_max < best_max:
                best_max = u_max
                best_u = u.copy()
            sgn = 1.0 if u[k_max] >= 0.0 else -1.0
            g = sgn * null_basis[k_max, :]
            g_norm = np.sqrt(np.dot(g, g))
            if g_norm <= 1e-12:
                break
            step = (0.4 * best_max / np.sqrt(float(it + 1))) / g_norm
            z = z - step * g
            u = u_base + null_basis @ z
        u = best_u

    u_rw = u + null_proj @ u_0
    return saturate_wheels(u_rw, max_torque)


@dataclass(slots=True)
class LInfinityAllocator(Allocator):
    """Minimum-L-infinity-norm (min-max wheel torque) allocator."""

    kind: ClassVar[int] = ALLOC_LINFINITY
