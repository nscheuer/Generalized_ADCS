r"""
Framing for the ADCS visualization stream, protocol version 1.

Every message is a 20-byte little-endian header followed by a payload::

    offset  size  field
    0       4     magic  b"ADVZ"
    4       2     protocol major version (uint16) = 1
    6       2     message type (uint16)
    8       4     payload length in bytes (uint32)
    12      8     sequence number (uint64), per sender, starting at 0

Rows and blocks carry the same float64 values, in the same channel order, as
the files of an ``adcs-vizlog`` (see ``docs/source/viz/protocol_v1.rst``).
Only the standard library and numpy are used, so this module is also the
reference for ports to other languages.
"""

__all__ = [
    "MAGIC",
    "PROTOCOL_VERSION",
    "HEADER",
    "MsgType",
    "ProtocolError",
    "pack_frame",
    "read_frame",
    "encode_json",
    "decode_json",
    "encode_row",
    "decode_row",
    "encode_block",
    "decode_block",
    "MAX_PAYLOAD",
]

import json
import socket
import struct
from enum import IntEnum
from typing import Any, List, Sequence, Tuple

import numpy as np

from ADCS.viz.writer import _jsonable

MAGIC = b"ADVZ"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("<4sHHIQ")
#: Upper bound on a single payload; blocks are split to stay below it.
MAX_PAYLOAD = 64 * 1024 * 1024


class MsgType(IntEnum):
    """Message types. Server = the Python side, client = the viewer."""

    HELLO = 1    #: server -> client, JSON {protocol, mode, manifest, scene}
    BLOCK = 2    #: server -> client, many rows of one run (playback)
    ROW = 3      #: server -> client, one row of one run (live)
    END = 4      #: server -> client, JSON {"run_id": id} or {} for end of session
    ACK = 5      #: client -> server, acknowledges header seq (lockstep mode)
    CONNECT = 6  #: client -> server, JSON {token, client, protocol}
    ERROR = 7    #: either direction, JSON {"message"}; the sender then closes


class ProtocolError(RuntimeError):
    """Malformed or unexpected message."""


def pack_frame(msg_type: MsgType, payload: bytes = b"", seq: int = 0) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD}")
    return HEADER.pack(MAGIC, PROTOCOL_VERSION, int(msg_type), len(payload), seq) + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed")
        got += k
    return bytes(buf)


def read_frame(sock: socket.socket) -> Tuple[MsgType, int, bytes]:
    r"""
    Read one message.

    :return: ``(type, seq, payload)``.
    :raises ConnectionError: If the peer closed the connection.
    :raises ProtocolError: On a bad magic, version or type.
    """
    magic, version, msg_type, length, seq = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"payload length {length} exceeds {MAX_PAYLOAD}")
    try:
        kind = MsgType(msg_type)
    except ValueError as e:
        raise ProtocolError(f"unknown message type {msg_type}") from e
    return kind, seq, _recv_exact(sock, length) if length else b""


def encode_json(obj: Any) -> bytes:
    return json.dumps(_jsonable(obj), allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def _encode_run_id(run_id: str) -> bytes:
    raw = str(run_id).encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _decode_run_id(payload: bytes) -> Tuple[str, int]:
    (n,) = struct.unpack_from("<H", payload, 0)
    return payload[2:2 + n].decode("utf-8"), 2 + n


def encode_row(run_id: str, values: np.ndarray) -> bytes:
    r"""
    ROW payload: ``u16 len, run_id (UTF-8), float64[row_width]``.

    ``values`` is one row with all channels concatenated in manifest order,
    i.e. byte-identical to the corresponding row of the vizlog files.
    """
    return _encode_run_id(run_id) + np.ascontiguousarray(values, dtype="<f8").tobytes()


def decode_row(payload: bytes) -> Tuple[str, np.ndarray]:
    run_id, off = _decode_run_id(payload)
    return run_id, np.frombuffer(payload, dtype="<f8", offset=off).copy()


def encode_block(run_id: str, row_start: int, columns: Sequence[np.ndarray]) -> bytes:
    r"""
    BLOCK payload: ``u16 len, run_id, u32 row_start, u32 n_rows``, then each
    channel in manifest order as ``float64[n_rows, width]`` (row-major), i.e. a
    slice of each channel file, concatenated.
    """
    n = len(columns[0]) if columns else 0
    parts: List[bytes] = [_encode_run_id(run_id), struct.pack("<II", row_start, n)]
    for col in columns:
        col = np.asarray(col, dtype="<f8")
        if len(col) != n:
            raise ProtocolError("block columns have different lengths")
        parts.append(np.ascontiguousarray(col).tobytes())
    return b"".join(parts)


def decode_block(payload: bytes, widths: Sequence[int]) -> Tuple[str, int, List[np.ndarray]]:
    run_id, off = _decode_run_id(payload)
    row_start, n = struct.unpack_from("<II", payload, off)
    off += 8
    cols = []
    for w in widths:
        count = n * w
        cols.append(np.frombuffer(payload, dtype="<f8", count=count, offset=off).reshape(n, w).copy())
        off += 8 * count
    if off != len(payload):
        raise ProtocolError(f"block has {len(payload) - off} trailing bytes")
    return run_id, row_start, cols
