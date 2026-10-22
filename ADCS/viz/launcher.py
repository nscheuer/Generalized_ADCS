__all__ = ["viewer_command", "launch_viewer", "view"]

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from ADCS.viz.binary_manager import find_viewer
from ADCS.viz.reader import VizLog
from ADCS.viz.stream import VizServer


def viewer_command(binary: str | Path, host: str, port: int, token: str,
                   extra_args: Sequence[str] = ()) -> List[str]:
    r"""
    Command line that starts the viewer and points it at a server.

    Godot passes arguments after ``--`` to the project
    (``OS.get_cmdline_user_args()``), so the viewer options follow it.
    """
    return [str(binary), "--", "--connect", f"{host}:{port}", "--token", token, *extra_args]


def launch_viewer(binary: str | Path, host: str, port: int, token: str,
                  extra_args: Sequence[str] = ()) -> subprocess.Popen:
    r"""
    Start the viewer as a child process.

    Inside WSL a Windows ``.exe`` is started through WSL interop and connects
    back to ``127.0.0.1``, which Windows forwards into WSL; no wider binding is
    needed.
    """
    return subprocess.Popen(viewer_command(binary, host, port, token, extra_args))


def view(
    log_path: str | Path,
    binary: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    launch: bool = True,
    extra_args: Sequence[str] = (),
) -> int:
    r"""
    Serve a recorded vizlog and open the viewer on it. Blocks until the viewer
    exits (or, with ``launch=False``, until interrupted).

    :param log_path: vizlog directory.
    :param binary: Viewer executable; resolved with :func:`~ADCS.viz.binary_manager.find_viewer` if omitted.
    :param host: Interface to serve on.
    :param port: Port; ``0`` picks a free one.
    :param launch: Start the viewer. ``False`` only serves and prints the
        arguments to give a viewer started by hand (e.g. from the Godot editor).
    :param extra_args: Extra viewer arguments.
    :return: The viewer's exit code (0 when not launched).
    """
    log = VizLog.open(log_path)
    exe = find_viewer(binary) if launch else None
    with VizServer(host=host, port=port) as server:
        server.serve_log(log)
        h, p = server.address
        if exe is None:
            print(f"Serving {log.path} on {h}:{p}. Start the viewer with:\n"
                  f"  -- --connect {h}:{p} --token {server.token}\n"
                  "Press Ctrl+C to stop.")
            try:
                while True:
                    server.wait_for_client(timeout=3600)
            except KeyboardInterrupt:
                return 0
        proc = launch_viewer(exe, h, p, server.token, extra_args)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()
