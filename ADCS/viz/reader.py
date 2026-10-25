__all__ = ["VizLog"]

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ADCS.viz.schema import FORMAT, SCHEMA_VERSION, Channel


class VizLog:
    r"""
    Read an ``adcs-vizlog`` directory.

    Channels are memory-mapped by default, so opening a large log is cheap and
    only the columns that are accessed are paged in.

    If a run was not completed (the writer crashed or is still appending), the
    number of usable rows is the smallest number of whole rows present across
    all of the run's channel files.

    :param path: vizlog directory.
    :param mmap: Memory-map channel files (read-only) instead of loading them.
    :raises ValueError: If the directory is not a vizlog of a supported version.
    """

    def __init__(self, path: str | Path, mmap: bool = True) -> None:
        self.path = Path(path)
        with open(self.path / "manifest.json", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{self.path} is not an {FORMAT} directory")
        version = self.manifest.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported vizlog schema version {version} (reader supports {SCHEMA_VERSION})")
        with open(self.path / "scene.json", encoding="utf-8") as f:
            self.scene: Dict[str, Any] = json.load(f)
        self.channels: Dict[str, Channel] = {
            c["name"]: Channel.from_json(c) for c in self.manifest["channels"]
        }
        self.mmap = mmap

    @classmethod
    def open(cls, path: str | Path, mmap: bool = True) -> "VizLog":
        return cls(path, mmap=mmap)

    @property
    def run_ids(self) -> List[str]:
        return [r["id"] for r in self.manifest["runs"]]

    @property
    def epoch_jd_tt(self) -> float:
        e = self.manifest["epoch"]
        return e["jd_tt_whole"] + e["jd_tt_frac"]

    def _run_dir(self, run_id: int | str) -> Path:
        rid = str(run_id)
        if rid not in self.run_ids:
            raise KeyError(f"no run {rid!r}; available: {self.run_ids}")
        return self.path / "runs" / rid

    def is_complete(self, run_id: int | str) -> bool:
        with open(self._run_dir(run_id) / "rows.json", encoding="utf-8") as f:
            return bool(json.load(f).get("complete", False))

    def n_rows(self, run_id: int | str) -> int:
        r"""Number of whole rows available in a run."""
        run_dir = self._run_dir(run_id)
        with open(run_dir / "rows.json", encoding="utf-8") as f:
            rows = json.load(f)
        if rows.get("complete"):
            return int(rows["n_rows"])
        # Only an incomplete run needs the (slow on network/WSL drives) file scan.
        return int(min(
            (run_dir / f"{c.name}.f64").stat().st_size // (8 * c.width)
            for c in self.channels.values()
        ))

    def read(self, run_id: int | str, name: str) -> np.ndarray:
        r"""
        One channel of one run.

        :return: Array of shape ``(n_rows, width)`` (read-only when memory-mapped).
        """
        if name not in self.channels:
            raise KeyError(f"no channel {name!r}")
        width = self.channels[name].width
        n = self.n_rows(run_id)
        file = self._run_dir(run_id) / f"{name}.f64"
        if n == 0:
            return np.empty((0, width), dtype="<f8")
        if self.mmap:
            return np.memmap(file, dtype="<f8", mode="r", shape=(n, width))
        return np.fromfile(file, dtype="<f8", count=n * width).reshape(n, width)

    def __contains__(self, name: str) -> bool:
        return name in self.channels

    def __repr__(self) -> str:
        return f"VizLog({str(self.path)!r}, runs={self.run_ids}, channels={list(self.channels)})"
