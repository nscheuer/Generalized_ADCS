__all__ = ["VizLogWriter", "make_manifest"]

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ADCS.helpers.animations.godot.schema import (
    FORMAT,
    QUATERNION_CONVENTION,
    REQUIRED_CHANNELS,
    SCHEMA_VERSION,
    Channel,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def make_manifest(
    channels: Iterable[Channel],
    epoch_jd_tt: float,
    epoch_iso_utc: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""
    Build the ``manifest.json`` content, without runs.

    Shared by :class:`VizLogWriter` and the live stream, so a streamed session
    describes itself exactly as a file would.

    :param channels: Channels every run contains.
    :param epoch_jd_tt: Julian date (TT) that ``t_s = 0`` refers to.
    :param epoch_iso_utc: Same instant as ISO-8601 UTC, for humans.
    :param metadata: Extra manifest entries.
    :return: Manifest dictionary with an empty ``runs`` list.
    """
    whole = float(np.floor(epoch_jd_tt))
    manifest: Dict[str, Any] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "epoch": {"jd_tt_whole": whole, "jd_tt_frac": float(epoch_jd_tt) - whole,
                  "iso_utc": epoch_iso_utc},
        "time_scale": "TT",
        "frames": {
            "GCRS": "geocentric, ICRS-aligned axes (Skyfield 'ICRF' at Earth centre)",
            "ITRS": "Earth-fixed",
            "BODY": "spacecraft body frame",
        },
        "quaternion_convention": QUATERNION_CONVENTION,
        "channels": [c.to_json() for c in channels],
        "runs": [],
    }
    if metadata:
        manifest.update(metadata)
    return manifest


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


class VizLogWriter:
    r"""
    Write an ``adcs-vizlog`` directory (format version 1).

    Layout::

        <path>/manifest.json            metadata, channel table, runs
        <path>/scene.json               static spacecraft / scene description
        <path>/runs/<id>/<channel>.f64  little-endian float64, shape (n_rows, width)
        <path>/runs/<id>/rows.json      {"n_rows": N, "complete": bool}

    Every run contains every declared channel; missing values are written as
    NaN. Rows can be written in bulk (:meth:`write_run`) or appended as they
    are produced (:meth:`append`), in which case ``rows.json`` is rewritten
    atomically at each flush so that a reader of an interrupted log sees only
    whole rows.

    :param path: Output directory. Created if missing.
    :param channels: Channels every run will contain. Must include all
        :data:`~ADCS.helpers.animations.godot.schema.REQUIRED_CHANNELS`.
    :param epoch_jd_tt: Julian date (TT) that ``t_s = 0`` refers to.
    :param epoch_iso_utc: Same instant as an ISO-8601 UTC string, for humans.
    :param scene: Static scene description written to ``scene.json``.
    :param metadata: Extra manifest entries (``generator``, ``ephemeris``,
        ``eop``, ...).
    :param overwrite: Replace an existing vizlog at ``path``.
    :param flush_every: Rows buffered per run before :meth:`append` flushes.
    """

    def __init__(
        self,
        path: str | Path,
        channels: Iterable[Channel],
        epoch_jd_tt: float,
        epoch_iso_utc: Optional[str] = None,
        scene: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        flush_every: int = 256,
    ) -> None:
        self.path = Path(path)
        self.channels: List[Channel] = list(channels)
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel names")
        missing = [n for n in REQUIRED_CHANNELS if n not in names]
        if missing:
            raise ValueError(f"missing required channels: {missing}")
        self._by_name = {c.name: c for c in self.channels}

        if self.path.exists():
            if not overwrite:
                raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
            if not (self.path / "manifest.json").exists() and any(self.path.iterdir()):
                raise FileExistsError(f"{self.path} is a non-empty directory that is not a vizlog")
            shutil.rmtree(self.path)
        (self.path / "runs").mkdir(parents=True)

        self.manifest = make_manifest(self.channels, epoch_jd_tt, epoch_iso_utc, metadata)
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._buffers: Dict[str, List[Dict[str, np.ndarray]]] = {}
        self._flush_every = int(flush_every)
        self._closed = False

        _write_json_atomic(self.path / "scene.json", scene or {})
        self._write_manifest()

    # ------------------------------------------------------------------
    def begin_run(self, run_id: int | str, dt_s: Optional[float] = None, **info: Any) -> str:
        r"""
        Start a run that rows will be appended to.

        :param run_id: Run identifier (``[A-Za-z0-9_-]``, at most 64 characters).
        :param dt_s: Nominal sample spacing [s], informational.
        :param info: Extra per-run manifest entries (e.g. ``seed``).
        :return: The run id as a string.
        """
        rid = str(run_id)
        if not _RUN_ID.match(rid):
            raise ValueError(f"invalid run id {rid!r}")
        if rid in self._runs:
            raise ValueError(f"run {rid!r} already exists")
        run_dir = self.path / "runs" / rid
        run_dir.mkdir()
        for c in self.channels:
            (run_dir / f"{c.name}.f64").touch()
        self._runs[rid] = {"id": rid, "n_rows": 0, "dt_s": dt_s, **info}
        self._buffers[rid] = []
        self._write_rows(rid, complete=False)
        self._write_manifest()
        return rid

    def append(self, run_id: int | str, **row: Any) -> None:
        r"""
        Append one row to a run started with :meth:`begin_run`.

        :param run_id: Run identifier.
        :param row: One value per channel name; omitted channels are NaN.
        """
        rid = str(run_id)
        unknown = set(row) - set(self._by_name)
        if unknown:
            raise KeyError(f"unknown channels {sorted(unknown)}")
        values = {}
        for c in self.channels:
            v = row.get(c.name)
            # Copy: callers may reuse one row buffer between appends.
            arr = np.full(c.width, np.nan) if v is None else np.array(v, dtype="<f8").reshape(-1)
            if arr.size != c.width:
                raise ValueError(f"channel {c.name!r} expects {c.width} values, got {arr.size}")
            values[c.name] = arr
        self._buffers[rid].append(values)
        if len(self._buffers[rid]) >= self._flush_every:
            self.flush(rid)

    def flush(self, run_id: Optional[int | str] = None) -> None:
        r"""Write buffered rows of one run (or all runs) to disk."""
        rids = list(self._buffers) if run_id is None else [str(run_id)]
        for rid in rids:
            rows = self._buffers[rid]
            if not rows:
                continue
            run_dir = self.path / "runs" / rid
            for c in self.channels:
                block = np.stack([r[c.name] for r in rows]).astype("<f8", copy=False)
                with open(run_dir / f"{c.name}.f64", "ab") as f:
                    f.write(np.ascontiguousarray(block).tobytes())
            self._runs[rid]["n_rows"] += len(rows)
            self._buffers[rid] = []
            self._write_rows(rid, complete=False)

    def end_run(self, run_id: int | str) -> None:
        r"""Flush a run and mark it complete."""
        rid = str(run_id)
        self.flush(rid)
        self._write_rows(rid, complete=True)
        self._write_manifest()

    def write_run(
        self,
        run_id: int | str,
        columns: Dict[str, np.ndarray],
        dt_s: Optional[float] = None,
        **info: Any,
    ) -> None:
        r"""
        Write a complete run from whole columns.

        :param run_id: Run identifier.
        :param columns: ``{channel: array}`` with shape ``(n_rows, width)`` (or
            ``(n_rows,)`` for width 1). Omitted channels are NaN.
        :param dt_s: Nominal sample spacing [s], informational.
        :param info: Extra per-run manifest entries.
        """
        unknown = set(columns) - set(self._by_name)
        if unknown:
            raise KeyError(f"unknown channels {sorted(unknown)}")
        lengths = {len(np.asarray(v)) for v in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"columns have different lengths: {sorted(lengths)}")
        n = lengths.pop()
        rid = self.begin_run(run_id, dt_s=dt_s, **info)
        run_dir = self.path / "runs" / rid
        for c in self.channels:
            if c.name in columns:
                block = np.asarray(columns[c.name], dtype="<f8").reshape(n, -1)
                if block.shape[1] != c.width:
                    raise ValueError(f"channel {c.name!r} expects width {c.width}, got {block.shape[1]}")
            else:
                block = np.full((n, c.width), np.nan, dtype="<f8")
            with open(run_dir / f"{c.name}.f64", "wb") as f:
                f.write(np.ascontiguousarray(block).tobytes())
        self._runs[rid]["n_rows"] = n
        self._write_rows(rid, complete=True)
        self._write_manifest()

    def close(self) -> None:
        r"""Flush and complete all runs. Safe to call twice."""
        if self._closed:
            return
        for rid in self._runs:
            self.end_run(rid)
        self._closed = True

    def __enter__(self) -> "VizLogWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Leave the partial log readable but marked incomplete.
            self.flush()
            self._write_manifest()

    # ------------------------------------------------------------------
    def _write_rows(self, rid: str, complete: bool) -> None:
        _write_json_atomic(self.path / "runs" / rid / "rows.json",
                           {"n_rows": self._runs[rid]["n_rows"], "complete": complete})

    def _write_manifest(self) -> None:
        self.manifest["runs"] = list(self._runs.values())
        _write_json_atomic(self.path / "manifest.json", self.manifest)
