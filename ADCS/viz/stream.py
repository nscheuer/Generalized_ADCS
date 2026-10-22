__all__ = ["VizServer"]

import secrets
import socket
import threading
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ADCS.viz.protocol import (
    MAX_PAYLOAD,
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    decode_json,
    encode_asset,
    encode_block,
    encode_json,
    encode_row,
    pack_frame,
    read_frame,
)
from ADCS.viz.reader import VizLog

_MODES = ("broadcast", "lockstep")


class VizServer:
    r"""
    TCP server the 3-D viewer connects to.

    The Python side always listens and the viewer always connects, which works
    the same way for a viewer on the same machine and for a Windows viewer
    talking to a simulation inside WSL2.

    A connecting viewer first sends ``CONNECT`` with the session token; a wrong
    token or protocol version gets an ``ERROR`` and is disconnected. It then
    receives ``HELLO`` (manifest and scene) followed by:

    - **playback** (:meth:`serve_log`): the whole log as ``BLOCK`` messages and
      an ``END`` per run, then ``END`` for the session. Every viewer that
      connects gets the full log.
    - **live** (:meth:`begin_live` / :meth:`publish_row`): ``ROW`` messages as
      the simulation produces them.

      * ``broadcast`` (default): rows go to a bounded queue drained by a
        background thread; if the viewer is slow or absent the oldest rows are
        dropped, so the simulation never waits. Rows produced before the viewer
        connects are kept up to ``queue_size``.
      * ``lockstep``: each row is sent synchronously and the simulation waits
        for the viewer's ``ACK`` (for deterministic capture). If the viewer
        disconnects or stops acknowledging, streaming stops with a warning and
        the simulation continues.

    One live viewer is served at a time.

    :param host: Interface to bind. ``127.0.0.1`` keeps the stream local.
    :param port: TCP port; ``0`` picks a free one (see :attr:`address`).
    :param token: Session token the viewer must present; random if ``None``.
    :param mode: ``"broadcast"`` or ``"lockstep"`` (live sessions).
    :param queue_size: Maximum rows buffered in broadcast mode.
    :param ack_timeout: Seconds to wait for an ``ACK`` in lockstep mode.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        token: Optional[str] = None,
        mode: str = "broadcast",
        queue_size: int = 4096,
        ack_timeout: float = 10.0,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.token = token if token is not None else secrets.token_hex(16)
        self.ack_timeout = float(ack_timeout)

        self._listener = socket.create_server((host, port))
        self._listener.settimeout(0.2)
        self.address: Tuple[str, int] = self._listener.getsockname()[:2]

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._hello: Optional[Dict[str, Any]] = None
        self._assets: List[bytes] = []
        self._log: Optional[VizLog] = None
        self._client: Optional[socket.socket] = None
        self._queue: Deque[bytes] = deque(maxlen=int(queue_size))
        self._dropped = 0
        self._seq = 0
        self._closing = False
        self._live_failed = False

        self._accept_thread = threading.Thread(target=self._accept_loop, name="VizServer-accept", daemon=True)
        self._accept_thread.start()
        self._sender_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._client is not None

    @property
    def dropped_rows(self) -> int:
        """Rows discarded in broadcast mode because the queue was full."""
        return self._dropped

    def wait_for_client(self, timeout: Optional[float] = None) -> bool:
        r"""Block until a viewer has connected (and received ``HELLO``)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._client is not None or self._closing, timeout)

    def serve_log(self, log: VizLog) -> None:
        r"""Serve a recorded log to every viewer that connects (playback)."""
        with self._lock:
            if self._hello is not None:
                raise RuntimeError("server already has a session")
            manifest = dict(log.manifest)
            self._log = log
            self._hello = {"protocol": PROTOCOL_VERSION, "mode": "playback",
                           "manifest": manifest, "scene": log.scene}
            self._assets = _scene_assets(log.scene)

    def begin_live(self, manifest: Dict[str, Any], scene: Dict[str, Any]) -> None:
        r"""
        Start a live session. Rows are then sent with :meth:`publish_row`.

        :param manifest: Manifest as built by :func:`~ADCS.viz.writer.make_manifest`.
        :param scene: Scene description.
        """
        with self._cond:
            if self._hello is not None:
                raise RuntimeError("server already has a session")
            self._hello = {"protocol": PROTOCOL_VERSION, "mode": f"live-{self.mode}",
                           "manifest": manifest, "scene": scene}
            self._assets = _scene_assets(scene)
            self._cond.notify_all()
        if self.mode == "broadcast":
            self._sender_thread = threading.Thread(target=self._sender_loop, name="VizServer-send", daemon=True)
            self._sender_thread.start()

    def publish_row(self, run_id: str, values: np.ndarray) -> None:
        r"""
        Send one row (all channels concatenated in manifest order).

        Never raises because of the viewer: in broadcast mode a full queue drops
        its oldest row; in lockstep mode a lost viewer ends streaming.
        """
        payload = encode_row(run_id, values)
        if self.mode == "broadcast":
            self._enqueue(MsgType.ROW, payload)
        else:
            self._send_lockstep(MsgType.ROW, payload)

    def end_run(self, run_id: str) -> None:
        self._end({"run_id": str(run_id)})

    def end_session(self) -> None:
        self._end({})

    def close(self) -> None:
        r"""Flush pending rows (briefly), then stop serving."""
        if self._sender_thread is not None:
            with self._cond:
                self._cond.wait_for(lambda: not self._queue or self._client is None, timeout=2.0)
        with self._cond:
            self._closing = True
            client, self._client = self._client, None
            self._cond.notify_all()
        if client is not None:
            _close_quietly(client)
        self._accept_thread.join(timeout=2.0)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)
        _close_quietly(self._listener)

    def __enter__(self) -> "VizServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --------------------------------------------------------------- internals
    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _end(self, info: Dict[str, Any]) -> None:
        payload = encode_json(info)
        if self.mode == "broadcast":
            self._enqueue(MsgType.END, payload)
        else:
            self._send_lockstep(MsgType.END, payload, expect_ack=False)

    def _enqueue(self, msg_type: MsgType, payload: bytes) -> None:
        with self._cond:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            self._queue.append(pack_frame(msg_type, payload, self._next_seq()))
            # Waking the sender with nobody to send to only steals the GIL from
            # the simulation; it is woken when a viewer connects instead.
            if self._client is not None:
                self._cond.notify_all()

    def _send_lockstep(self, msg_type: MsgType, payload: bytes, expect_ack: bool = True) -> None:
        with self._lock:
            client = self._client
            if client is None or self._live_failed:
                return
            seq = self._next_seq()
            try:
                client.sendall(pack_frame(msg_type, payload, seq))
                if expect_ack:
                    client.settimeout(self.ack_timeout)
                    while True:
                        kind, ack_seq, _ = read_frame(client)
                        if kind == MsgType.ACK and ack_seq == seq:
                            break
                        if kind == MsgType.ERROR:
                            raise ConnectionError("viewer reported an error")
            except (OSError, ConnectionError, ProtocolError) as e:
                self._live_failed = True
                self._client = None
                _close_quietly(client)
                warnings.warn(f"VizServer: viewer lost in lockstep mode ({e}); streaming stopped", stacklevel=4)

    def _sender_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closing or (self._queue and self._client is not None))
                if self._closing:
                    return
                # Drain everything queued in one send: one wake-up per batch,
                # not per row.
                batch = b"".join(self._queue)
                self._queue.clear()
                self._cond.notify_all()
                client = self._client
            try:
                client.sendall(batch)
            except OSError:
                with self._cond:
                    if self._client is client:
                        self._client = None
                        self._cond.notify_all()
                _close_quietly(client)

    def _accept_loop(self) -> None:
        while not self._closing:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), name="VizServer-conn", daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            conn.settimeout(10.0)
            kind, _, payload = read_frame(conn)
            info = decode_json(payload) if kind == MsgType.CONNECT else {}
            if kind != MsgType.CONNECT or info.get("token") != self.token:
                return self._reject(conn, "expected CONNECT with a valid token")
            if info.get("protocol") != PROTOCOL_VERSION:
                return self._reject(conn, f"protocol {info.get('protocol')} not supported (server speaks {PROTOCOL_VERSION})")
            conn.settimeout(None)

            with self._cond:
                self._cond.wait_for(lambda: self._hello is not None or self._closing)
                if self._closing:
                    return _close_quietly(conn)
                hello = self._hello
                log = self._log
            if log is not None:
                return self._serve_playback(conn, hello, log)

            with self._cond:
                if self._client is not None or self._live_failed:
                    return self._reject(conn, "a viewer is already connected to this live session")
                conn.sendall(pack_frame(MsgType.HELLO, encode_json(hello), self._next_seq()))
                for asset in self._assets:
                    conn.sendall(pack_frame(MsgType.ASSET, asset, self._next_seq()))
                self._client = conn
                self._cond.notify_all()
        except (OSError, ConnectionError, ProtocolError, ValueError):
            _close_quietly(conn)

    def _serve_playback(self, conn: socket.socket, hello: Dict[str, Any], log: VizLog) -> None:
        # Playback messages use their own sequence numbers per connection.
        seq = 0

        def send(kind: MsgType, payload: bytes) -> None:
            nonlocal seq
            conn.sendall(pack_frame(kind, payload, seq))
            seq += 1

        try:
            send(MsgType.HELLO, encode_json(hello))
            for asset in self._assets:
                send(MsgType.ASSET, asset)
            names = list(log.channels)
            row_bytes = 8 * sum(c.width for c in log.channels.values())
            rows_per_block = max(1, (MAX_PAYLOAD // 4) // max(row_bytes, 1))
            for rid in log.run_ids:
                n = log.n_rows(rid)
                cols = [log.read(rid, name) for name in names]
                for start in range(0, n, rows_per_block):
                    stop = min(n, start + rows_per_block)
                    send(MsgType.BLOCK, encode_block(rid, start, [c[start:stop] for c in cols]))
                send(MsgType.END, encode_json({"run_id": rid}))
            send(MsgType.END, encode_json({}))
            # Keep the connection open until the viewer closes it.
            while not self._closing:
                conn.settimeout(0.5)
                try:
                    read_frame(conn)
                except socket.timeout:
                    continue
        except (OSError, ConnectionError, ProtocolError):
            pass
        finally:
            _close_quietly(conn)

    def _reject(self, conn: socket.socket, message: str) -> None:
        try:
            conn.sendall(pack_frame(MsgType.ERROR, encode_json({"message": message})))
        except OSError:
            pass
        _close_quietly(conn)


def _close_quietly(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _scene_assets(scene: Dict[str, Any]) -> List[bytes]:
    """ASSET payloads for the files a scene refers to (spacecraft model levels of detail)."""
    out = []
    mesh = (scene or {}).get("mesh") or {}
    for i, lod in enumerate(mesh.get("lods", [])):
        path = Path(lod.get("path", ""))
        if not path.is_file():
            warnings.warn(f"VizServer: scene refers to missing model file {path}", stacklevel=3)
            continue
        data = path.read_bytes()
        if len(data) > MAX_PAYLOAD - 4096:
            warnings.warn(f"VizServer: {path.name} is too large to stream ({len(data)} bytes)", stacklevel=3)
            continue
        meta = {"kind": "model_lod", "lod": i, "sha256": lod.get("sha256"), "triangles": lod.get("triangles")}
        out.append(encode_asset(path.name, meta, data))
    return out
