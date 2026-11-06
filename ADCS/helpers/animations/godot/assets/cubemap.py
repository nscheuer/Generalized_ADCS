r"""
Equirectangular maps -> cube maps, with the face convention of OpenGL/Vulkan
cube textures, so a shader samples the result with the raw body-fixed
direction (ITRS for Earth): ``texture(cube, n_itrs)``.

Face order and orientation (``s``, ``t`` in [-1, 1] at texel centres, ``s``
along image columns left to right, ``t`` along rows top to bottom):

====  =====================
face  direction (x, y, z)
====  =====================
+X    ( 1, -t, -s)
-X    (-1, -t,  s)
+Y    ( s,  1,  t)
-Y    ( s, -1, -t)
+Z    ( s, -t,  1)
-Z    (-s, -t, -1)
====  =====================

Equirectangular sources span longitude -180..180 (left to right) and latitude
90..-90 (top to bottom), pixel centres at half-pixel offsets; latitude is the
geocentric angle on the sphere.
"""

__all__ = ["FACES", "face_directions", "directions_to_face", "lonlat", "sample_equirect",
           "equirect_to_cube"]

from typing import Callable, Iterator, Tuple

import numpy as np

FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def _face_dir(face: str, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    one = np.ones_like(s)
    x, y, z = {
        "+X": (one, -t, -s),
        "-X": (-one, -t, s),
        "+Y": (s, one, t),
        "-Y": (s, -one, -t),
        "+Z": (s, -t, one),
        "-Z": (-s, -t, -one),
    }[face]
    d = np.stack([x, y, z], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def face_directions(face: str, size: int, row0: int = 0, row1: int | None = None) -> np.ndarray:
    r"""Unit directions of texel centres for rows ``row0:row1`` of a face, shape ``(rows, size, 3)``."""
    row1 = size if row1 is None else row1
    c = (2.0 * (np.arange(size) + 0.5) / size - 1.0)
    s, t = np.meshgrid(c, c[row0:row1])
    return _face_dir(face, s, t)


def directions_to_face(d: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Inverse mapping: which face and texel a direction falls in.

    :return: ``(face_index, row, col)`` integer arrays (nearest texel).
    """
    d = np.asarray(d, dtype=float)
    ax = np.argmax(np.abs(d), axis=-1)
    sign = np.take_along_axis(d, ax[..., None], axis=-1)[..., 0] >= 0
    face = 2 * ax + (~sign).astype(int)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    m = np.abs(np.take_along_axis(d, ax[..., None], axis=-1)[..., 0])
    s = np.select([face == 0, face == 1, face == 2, face == 3, face == 4, face == 5],
                  [-z, z, x, x, x, -x]) / m
    t = np.select([face == 0, face == 1, face == 2, face == 3, face == 4, face == 5],
                  [-y, -y, z, -z, -y, -y]) / m
    col = np.clip(((s + 1.0) * 0.5 * size).astype(int), 0, size - 1)
    row = np.clip(((t + 1.0) * 0.5 * size).astype(int), 0, size - 1)
    return face, row, col


def lonlat(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Longitude and latitude [rad] of unit directions."""
    return np.arctan2(d[..., 1], d[..., 0]), np.arcsin(np.clip(d[..., 2], -1.0, 1.0))


def sample_equirect(img: np.ndarray, lon: np.ndarray, lat: np.ndarray, nearest: bool = False) -> np.ndarray:
    r"""
    Sample an equirectangular image (H, W[, C]) at longitudes/latitudes [rad].

    Bilinear by default (wrapping in longitude, clamped at the poles), or
    nearest-neighbour for categorical data such as masks. Returns float32.
    """
    h, w = img.shape[:2]
    x = (lon + np.pi) / (2.0 * np.pi) * w - 0.5
    y = (np.pi / 2.0 - lat) / np.pi * h - 0.5
    if nearest:
        xi = np.mod(np.rint(x).astype(np.int64), w)
        yi = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
        return img[yi, xi].astype(np.float32)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x0w, x1w = np.mod(x0, w), np.mod(x0 + 1, w)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = img[y0c, x0w].astype(np.float32) * (1 - fx) + img[y0c, x1w].astype(np.float32) * fx
    bot = img[y1c, x0w].astype(np.float32) * (1 - fx) + img[y1c, x1w].astype(np.float32) * fx
    return top * (1 - fy) + bot * fy


def equirect_to_cube(
    size: int,
    texel_fn: Callable[[np.ndarray], np.ndarray],
    block_rows: int = 256,
) -> Iterator[Tuple[str, np.ndarray]]:
    r"""
    Build cube faces block by block (bounded memory for large faces).

    :param size: Face edge in texels.
    :param texel_fn: Maps directions ``(rows, size, 3)`` to texel values
        ``(rows, size[, C])``.
    :param block_rows: Rows computed per block.
    :return: Iterator of ``(face_name, face_array)``.
    """
    for face in FACES:
        parts = []
        for r0 in range(0, size, block_rows):
            parts.append(texel_fn(face_directions(face, size, r0, min(size, r0 + block_rows))))
        yield face, np.concatenate(parts, axis=0)
