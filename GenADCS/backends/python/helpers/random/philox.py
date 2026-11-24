"""Counter-based Philox4x32-10 draws for compiled simulation kernels.

Counter words are (sample tick low, sample tick high, sensor index, block).
The 64-bit seed supplies the two key words.  Each stream is independent of
sampling order and of adaptive integrator trial steps.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _philox4x32(seed, tick, sensor, block):
    mask = np.uint64(0xffffffff)
    c0 = np.uint64(tick) & mask
    c1 = (np.uint64(tick) >> np.uint64(32)) & mask
    c2 = np.uint64(sensor) & mask
    c3 = np.uint64(block) & mask
    k0 = np.uint64(seed) & mask
    k1 = (np.uint64(seed) >> np.uint64(32)) & mask
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & mask, p1 & mask, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & mask, p0 & mask
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask
    return c0, c1, c2, c3


@njit(cache=True)
def standard_normals(seed, tick, sensor, count):
    """Generate normals from Philox words using a fixed Box-Muller mapping."""
    out = np.empty(count, dtype=np.float64)
    for block in range((count + 3) // 4):
        words = _philox4x32(seed, tick, sensor, block)
        for pair in range(2):
            offset = block * 4 + pair * 2
            if offset >= count:
                break
            u1 = (float(words[2 * pair]) + 0.5) * (1.0 / 4294967296.0)
            u2 = (float(words[2 * pair + 1]) + 0.5) * (1.0 / 4294967296.0)
            radius = math.sqrt(-2.0 * math.log(u1))
            angle = 2.0 * math.pi * u2
            out[offset] = radius * math.cos(angle)
            if offset + 1 < count:
                out[offset + 1] = radius * math.sin(angle)
    return out


def validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    return np.uint64(seed)


def next_draw(owner, count):
    """Advance the local stream for a standalone noise or sensor object."""
    tick = getattr(owner, "_random_tick", 0)
    draw = standard_normals(validate_seed(getattr(owner, "seed", 0)), tick, 0x40000000, count)
    owner._random_tick = tick + 1
    return draw
