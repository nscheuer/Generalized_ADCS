__all__ = ["viewer_command", "launch_viewer", "view", "viewer_project"]

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from ADCS.helpers.animations.godot.binary_manager import _windows_local_appdata, find_viewer, is_wsl
from ADCS.helpers.animations.godot.reader import VizLog
from ADCS.helpers.animations.godot.stream import VizServer
from ADCS.helpers.animations.godot.assets.fetch import cache_dir as visualization_asset_dir
from ADCS.helpers.animations.godot.assets.planets import ensure_planet_textures, missing_planet_textures

_REPO_VIEWER = Path(__file__).with_name("viewer")


def _is_windows_exe(binary: str | Path) -> bool:
    return str(binary).lower().endswith(".exe")


def _host_path(path: Path, binary: str | Path) -> str:
    """A path as the viewer process sees it (Windows form for a Windows viewer started from WSL)."""
    if is_wsl() and _is_windows_exe(binary):
        return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True,
                              check=True).stdout.strip()
    return str(path)


def viewer_project(binary: str | Path, project: Optional[str | Path] = None) -> Optional[Path]:
    r"""
    The Godot project to run when the "viewer" is a plain Godot binary
    (development), or ``None`` for an exported viewer.

    Order: ``project``, ``$ADCS_VIEWER_PROJECT``, then the repository's
    ``viewer/`` directory if the binary looks like a Godot editor build.
    Inside WSL with a Windows Godot, the project is copied to the Windows side
    (Godot cannot import a project over a ``\\\\wsl$`` path reliably).
    """
    chosen = project or os.environ.get("ADCS_VIEWER_PROJECT")
    if chosen is None and Path(binary).name.lower().startswith("godot") and (_REPO_VIEWER / "project.godot").exists():
        chosen = _REPO_VIEWER
    if chosen is None:
        return None
    src = Path(chosen).resolve()
    if is_wsl() and _is_windows_exe(binary) and str(src).startswith("/") and not str(src).startswith("/mnt/"):
        local = _windows_local_appdata()
        if local is not None:
            dest = local / "generalized_adcs" / "viewer_project"
            if dest.exists():
                # Keep Godot's import cache; replace everything else.
                for item in dest.iterdir():
                    if item.name != ".godot":
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
            shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".godot"))
            return dest
    return src


def viewer_command(binary: str | Path, host: str, port: int, token: str,
                   extra_args: Sequence[str] = (), project: Optional[Path] = None,
                   assets: Optional[Path] = None) -> List[str]:
    r"""
    Command line that starts the viewer and points it at a server.

    Godot passes arguments after ``--`` to the project
    (``OS.get_cmdline_user_args()``), so the viewer options follow it.

    :param project: Godot project directory, when ``binary`` is a plain Godot build.
    :param assets: Visualization asset cache (``python -m ADCS.helpers.animations.godot.assets path``).
    """
    cmd = [str(binary)]
    if project is not None:
        cmd += ["--path", _host_path(project, binary)]
    cmd += ["--", "--connect", f"{host}:{port}", "--token", token]
    if assets is not None:
        cmd += ["--assets", _host_path(assets, binary)]
        cmd += ["--model-request-file", _host_path(Path(assets) / ".spacecraft-model-request.json", binary),
                "--model-response-file", _host_path(Path(assets) / ".spacecraft-model-response.json", binary)]
    return cmd + list(extra_args)


def launch_viewer(binary: str | Path, host: str, port: int, token: str,
                  extra_args: Sequence[str] = (), project: Optional[Path] = None,
                  assets: Optional[Path] = None) -> subprocess.Popen:
    r"""
    Start the viewer as a child process.

    Inside WSL a Windows ``.exe`` is started through WSL interop and connects
    back to ``127.0.0.1``, which Windows forwards into WSL; no wider binding is
    needed.
    """
    return subprocess.Popen(viewer_command(binary, host, port, token, extra_args, project, assets))


def _stop(proc: subprocess.Popen) -> None:
    """Close the viewer if it is still running (also stops a Windows viewer started from WSL)."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class _ModelRequestBroker:
    """Resolve spacecraft model requests written by the viewer's dropdown."""

    def __init__(self, root: Path):
        self.root = root
        self.request = root / ".spacecraft-model-request.json"
        self.response = root / ".spacecraft-model-response.json"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="adcs-model-assets", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.request.unlink(missing_ok=True)
        self.response.unlink(missing_ok=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.request.unlink(missing_ok=True)
        self.response.unlink(missing_ok=True)

    def _run(self) -> None:
        last_id = ""
        while not self.stop_event.is_set():
            try:
                if self.request.is_file():
                    request = json.loads(self.request.read_text(encoding="utf-8"))
                    request_id = str(request.get("id", ""))
                    if request_id and request_id != last_id:
                        last_id = request_id
                        from ADCS.helpers.animations.godot.assets.spacecraft import ensure_model
                        name = str(request.get("model", ""))
                        try:
                            path = ensure_model(name, root=self.root)
                            result = {"id": request_id, "ok": path is not None}
                        except (OSError, RuntimeError, ValueError) as error:
                            print(f"NASA model download failed: {error}", flush=True)
                            result = {"id": request_id, "ok": False}
                        temporary = self.response.with_suffix(".tmp")
                        temporary.write_text(json.dumps(result), encoding="utf-8")
                        temporary.replace(self.response)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            self.stop_event.wait(0.1)


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """Treat SIGTERM like Ctrl+C, so being terminated still closes the viewer."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _default_assets() -> Optional[Path]:
    try:
        from ADCS.helpers.animations.godot.assets.fetch import cache_dir
    except ImportError:
        return None
    root = cache_dir()
    return root if (root / "processed" / "current.json").exists() or (root / "planet_textures").exists() else None


def view(
    log_path: str | Path,
    binary: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    launch: bool = True,
    extra_args: Sequence[str] = (),
    project: Optional[str | Path] = None,
    assets: Optional[str | Path] = None,
) -> int:
    r"""
    Serve a recorded vizlog and open the viewer on it. Blocks until the viewer
    exits (or, with ``launch=False``, until interrupted).

    :param log_path: vizlog directory.
    :param binary: Viewer executable; resolved with :func:`~ADCS.helpers.animations.godot.binary_manager.find_viewer` if omitted.
    :param host: Interface to serve on.
    :param port: Port; ``0`` picks a free one.
    :param launch: Start the viewer. ``False`` only serves and prints the
        arguments to give a viewer started by hand (e.g. from the Godot editor).
    :param extra_args: Extra viewer arguments.
    :param project: Godot project when ``binary`` is a plain Godot build (see :func:`viewer_project`).
    :param assets: Asset cache; defaults to the built cache of :mod:`ADCS.helpers.animations.godot.assets`, if any.
    :return: The viewer's exit code (0 when not launched).
    """
    log = VizLog.open(log_path)
    exe = find_viewer(binary) if launch else None
    asset_dir = Path(assets) if assets is not None else visualization_asset_dir()
    pending = any(missing_planet_textures(lod, root=asset_dir) for lod in ("2k", "8k")) if launch else False
    completion_file = asset_dir / f".textures-ready-{uuid.uuid4().hex}" if pending else None
    ready_file = asset_dir / f".viewer-ready-{uuid.uuid4().hex}" if pending else None
    if completion_file is not None:
        completion_file.parent.mkdir(parents=True, exist_ok=True)
        extra_args = (*extra_args, "--asset-prompt-file", str(completion_file),
                      "--asset-ready-file", str(ready_file))
    with VizServer(host=host, port=port) as server:
        server.serve_log(log)
        h, p = server.address
        if exe is None:
            print(f"Serving {log.path} on {h}:{p}. Start the viewer with:\n"
                  f"  -- --connect {h}:{p} --token {server.token}"
                  + (f" --assets {asset_dir}" if asset_dir else "") + "\nPress Ctrl+C to stop.")
            try:
                while True:
                    server.wait_for_client(timeout=3600)
            except KeyboardInterrupt:
                return 0
        model_broker = _ModelRequestBroker(asset_dir)
        model_broker.prepare()
        proc = launch_viewer(exe, h, p, server.token, extra_args, viewer_project(exe, project), asset_dir)
        with _sigterm_as_interrupt():
            try:
                if pending:
                    try:
                        deadline = time.monotonic() + 30.0
                        while proc.poll() is None and not ready_file.exists() and time.monotonic() < deadline:
                            time.sleep(0.05)
                        if missing_planet_textures("2k", root=asset_dir):
                            ensure_planet_textures("2k", root=asset_dir)
                        if missing_planet_textures("8k", root=asset_dir):
                            ensure_planet_textures("8k", root=asset_dir)
                    except (OSError, RuntimeError, ValueError) as error:
                        print(f"Texture preparation failed; continuing with available maps: {error}")
                    finally:
                        completion_file.touch()
                model_broker.start()
                return proc.wait()
            except KeyboardInterrupt:
                return 130
            finally:
                model_broker.stop()
                _stop(proc)
                if completion_file is not None:
                    completion_file.unlink(missing_ok=True)
                if ready_file is not None:
                    ready_file.unlink(missing_ok=True)
