__all__ = ["VizClient"]

import socket
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ADCS.viz.protocol import (
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    decode_block,
    decode_json,
    decode_row,
    encode_json,
    pack_frame,
    read_frame,
)


class VizClient:
    r"""
    Reference client for the visualization stream (what a viewer does).

    Used by the tests and as the model for viewer implementations: connect,
    present the token, read ``HELLO``, then consume ``BLOCK``/``ROW``/``END``
    messages, acknowledging rows when the session is in lockstep mode.

    :param host: Server host.
    :param port: Server port.
    :param token: Session token printed or passed by the launcher.
    :param name: Client name sent in ``CONNECT``.
    """

    def __init__(self, host: str, port: int, token: str, name: str = "python-reference") -> None:
        self.host, self.port, self.token, self.name = host, int(port), token, name
        self.sock: Optional[socket.socket] = None
        self.hello: Dict[str, Any] = {}
        self.names: List[str] = []
        self.widths: List[int] = []
        self._ack_seq = 0

    @property
    def lockstep(self) -> bool:
        return self.hello.get("mode") == "live-lockstep"

    def connect(self, timeout: float = 10.0) -> Dict[str, Any]:
        r"""
        Connect and read ``HELLO``.

        :return: The ``HELLO`` content (``protocol``, ``mode``, ``manifest``, ``scene``).
        :raises ConnectionRefusedError: If the server rejects the client.
        """
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(pack_frame(MsgType.CONNECT, encode_json(
            {"token": self.token, "client": self.name, "protocol": PROTOCOL_VERSION})))
        kind, _, payload = read_frame(self.sock)
        if kind == MsgType.ERROR:
            raise ConnectionRefusedError(decode_json(payload).get("message", "rejected"))
        if kind != MsgType.HELLO:
            raise ProtocolError(f"expected HELLO, got {kind.name}")
        self.hello = decode_json(payload)
        channels = self.hello["manifest"]["channels"]
        self.names = [c["name"] for c in channels]
        self.widths = [int(c["width"]) for c in channels]
        self.sock.settimeout(None)
        return self.hello

    def split_row(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        r"""Split a concatenated row into ``{channel: values}``."""
        out, i = {}, 0
        for name, w in zip(self.names, self.widths):
            out[name] = values[i:i + w]
            i += w
        if i != len(values):
            raise ProtocolError(f"row has {len(values)} values, manifest implies {i}")
        return out

    def messages(self, timeout: Optional[float] = None) -> Iterator[Tuple[MsgType, Any]]:
        r"""
        Yield decoded messages until the session ``END``.

        - ``(BLOCK, (run_id, row_start, {channel: array (n, width)}))``
        - ``(ROW, (run_id, {channel: array (width,)}))``
        - ``(END, run_id)`` for the end of a run

        :param timeout: Socket timeout per message (``None`` waits forever).
        """
        assert self.sock is not None, "connect() first"
        self.sock.settimeout(timeout)
        while True:
            kind, seq, payload = read_frame(self.sock)
            if kind == MsgType.BLOCK:
                rid, start, cols = decode_block(payload, self.widths)
                yield kind, (rid, start, dict(zip(self.names, cols)))
            elif kind == MsgType.ROW:
                rid, values = decode_row(payload)
                if self.lockstep:
                    self.sock.sendall(pack_frame(MsgType.ACK, b"", seq))
                yield kind, (rid, self.split_row(values))
            elif kind == MsgType.END:
                rid = decode_json(payload).get("run_id")
                if rid is None:
                    return
                yield kind, rid
            elif kind == MsgType.ERROR:
                raise ConnectionError(decode_json(payload).get("message", "server error"))
            else:
                raise ProtocolError(f"unexpected {kind.name}")

    def read_all(self, timeout: Optional[float] = None) -> Dict[str, Dict[str, np.ndarray]]:
        r"""
        Collect a whole session.

        :return: ``{run_id: {channel: array (n_rows, width)}}``.
        """
        parts: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for kind, data in self.messages(timeout=timeout):
            if kind == MsgType.BLOCK:
                rid, _, cols = data
                run = parts.setdefault(rid, {n: [] for n in self.names})
                for n, c in cols.items():
                    run[n].append(c)
            elif kind == MsgType.ROW:
                rid, cols = data
                run = parts.setdefault(rid, {n: [] for n in self.names})
                for n, c in cols.items():
                    run[n].append(c.reshape(1, -1))
        return {rid: {n: (np.vstack(v) if v else np.empty((0, w))) for (n, v), w in zip(cols.items(), self.widths)}
                for rid, cols in parts.items()}

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "VizClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
