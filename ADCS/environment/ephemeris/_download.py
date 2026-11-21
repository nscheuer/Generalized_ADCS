"""Locate bundled and downloaded environment kernels."""

from __future__ import annotations

from pathlib import Path


def kernel_path(filename: str) -> Path:
    """Return the environment kernel path, whether or not it exists yet."""
    return Path(__file__).resolve().parents[1] / "kernels" / filename
