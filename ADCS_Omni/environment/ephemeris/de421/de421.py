"""JPL DE421 planetary ephemeris read directly from its SPK binary kernel."""

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np

from ..ephemeris_model import (
    BODY_EARTH, BODY_MOON, BODY_SUN, EphemerisModel,
    planetary_position, planetary_velocity,
)


@lru_cache(maxsize=4)
def _load_spk(path_string):
    """Parse DAF/SPK descriptors and retain the Chebyshev segment data."""
    path = Path(path_string)
    raw = path.read_bytes()
    if len(raw) < 1024 or raw[:8] != b"DAF/SPK ":
        raise ValueError(f"not a DAF/SPK binary kernel: {path}")
    marker = raw[88:96]
    if marker == b"LTL-IEEE":
        endian = "<"
    elif marker == b"BIG-IEEE":
        endian = ">"
    else:
        raise ValueError(f"unsupported SPK binary format marker {marker!r}")

    nd, ni = struct.unpack_from(endian + "ii", raw, 8)
    if (nd, ni) != (2, 6):
        raise ValueError(f"unsupported DAF summary layout ND={nd}, NI={ni}")
    forward_record = struct.unpack_from(endian + "i", raw, 76)[0]
    summary_size = nd * 8 + ni * 4
    segments = []
    times = []
    arrays = []
    record_number = forward_record
    while record_number:
        record_offset = (record_number - 1) * 1024
        if record_offset + 1024 > len(raw):
            raise ValueError(f"invalid DAF summary record in {path}")
        next_record, _, count = struct.unpack_from(endian + "ddd", raw, record_offset)
        if count < 0 or count > (1024 - 24) // summary_size:
            raise ValueError(f"invalid summary count {count} in {path}")
        for i in range(int(count)):
            item_offset = record_offset + 24 + i * summary_size
            start_et, end_et = struct.unpack_from(endian + "dd", raw, item_offset)
            target, center, frame, segment_type, first_word, last_word = struct.unpack_from(
                endian + "6i", raw, item_offset + nd * 8
            )
            if segment_type not in (2, 3):
                continue
            first = first_word - 1
            last = last_word
            if first < 0 or last <= first or last * 8 > len(raw):
                raise ValueError(f"invalid SPK data addresses for target {target}")
            segment_data = np.frombuffer(raw, dtype=endian + "f8", count=last - first, offset=first * 8)
            if segment_data.size < 4:
                raise ValueError(f"truncated SPK segment for target {target}")
            offset = sum(array.size for array in arrays)
            arrays.append(np.asarray(segment_data, dtype=np.float64).copy())
            segments.append((target, center, segment_type, offset, segment_data.size))
            times.append((start_et, end_et))
        record_number = int(next_record)

    if not segments:
        raise ValueError(f"no supported type 2/3 SPK segments found in {path}")
    return (
        np.asarray(segments, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
        np.concatenate(arrays).astype(np.float64, copy=False),
    )


_BODY_INDICES = {
    "mercury": 0, "venus": 1, "earth-moon barycenter": 2, "emb": 2,
    "mars": 3, "jupiter": 4, "saturn": 5, "uranus": 6,
    "neptune": 7, "pluto": 8, "moon": 9, "sun": 10, "earth": 11,
}
_NAIF_TO_INDEX = {
    199: 0, 1: 0, 299: 1, 2: 1, 3: 2, 4: 3, 499: 3,
    5: 4, 599: 4, 6: 5, 699: 5, 7: 6, 799: 6,
    8: 7, 899: 7, 9: 8, 999: 8, 301: 9, 10: 10, 399: 11,
}


class DE421(EphemerisModel):
    """DE421 data for 1899-12-04 through 2053 (TDB Julian dates).

    Position and velocity methods return Earth-centered vectors in km and
    km/s. The bundled SPK type 2/3 Chebyshev coefficients are evaluated by
    Numba kernels; the implementation does not depend on Skyfield.
    """

    DATA_FILENAME = "de421.bsp"

    def __init__(self, filepath=None):
        path = Path(filepath).expanduser().resolve() if filepath is not None else self._packaged_ephemeris_path()
        super().__init__(*_load_spk(str(path)))

    @staticmethod
    def _packaged_ephemeris_path():
        return Path(__file__).resolve().parent / DE421.DATA_FILENAME

    @property
    def available_bodies(self):
        return tuple(sorted(_BODY_INDICES))

    @staticmethod
    def _planet_index(body):
        if isinstance(body, str):
            try:
                return _BODY_INDICES[body.lower()]
            except KeyError as error:
                raise ValueError(f"unsupported DE421 body {body!r}") from error
        body_id = int(body)
        if body_id in (BODY_SUN, BODY_EARTH, BODY_MOON):
            return {BODY_SUN: 10, BODY_EARTH: 11, BODY_MOON: 9}[body_id]
        try:
            return _NAIF_TO_INDEX[body_id]
        except KeyError as error:
            raise ValueError(f"unsupported DE421 body identifier {body_id}") from error

    def _relative_eci(self, kernel, target, epoch_jd_tdb, observer):
        target_index = self._planet_index(target)
        observer_index = self._planet_index(observer)
        epochs = np.asarray(epoch_jd_tdb, dtype=np.float64)
        if epochs.ndim == 0:
            return (kernel(target_index, float(epochs), self._segments, self._times, self._data)
                    - kernel(observer_index, float(epochs), self._segments, self._times, self._data))
        output = np.empty(epochs.shape + (3,), dtype=np.float64)
        for index in np.ndindex(epochs.shape):
            epoch = float(epochs[index])
            output[index] = (
                kernel(target_index, epoch, self._segments, self._times, self._data)
                - kernel(observer_index, epoch, self._segments, self._times, self._data)
            )
        return output

    def position_eci(self, target, epoch_jd_tdb, *, observer="earth"):
        """Return geometric target-minus-observer position [km] at TDB JD."""
        return self._relative_eci(planetary_position, target, epoch_jd_tdb, observer)

    def velocity_eci(self, target, epoch_jd_tdb, *, observer="earth"):
        """Return geometric target-minus-observer velocity [km/s] at TDB JD."""
        return self._relative_eci(planetary_velocity, target, epoch_jd_tdb, observer)

    def sun_position_eci(self, epoch_jd_tdb):
        return self.position(BODY_SUN, epoch_jd_tdb)


__all__ = ["DE421"]
