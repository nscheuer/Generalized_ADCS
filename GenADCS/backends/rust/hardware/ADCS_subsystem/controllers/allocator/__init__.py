"""Rust backend reaction-wheel torque allocators."""

from ..._subsystem import (
    Allocator,
    L2Allocator,
    LInfinityAllocator,
    allocate_l2,
    allocate_linfinity,
)

__all__ = [
    "Allocator",
    "L2Allocator",
    "LInfinityAllocator",
    "allocate_l2",
    "allocate_linfinity",
]
