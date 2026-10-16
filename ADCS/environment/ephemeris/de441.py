"""JPL DE441 planetary ephemeris from its two NAIF SPK files."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

from skyfield.api import load
from skyfield.jpllib import Stack, VectorSum

from .de421 import DE421
from ._download import download_with_progress


class _CombinedKernel:
    """Small Skyfield-compatible view over DE441's two time partitions."""

    def __init__(self, paths):
        r"""Load both DE441 BSP partitions as one logical kernel.

        :param paths: Paths to the two DE441 partition files.
        :type paths: tuple[pathlib.Path, pathlib.Path]
        :return: None
        :rtype: None
        """
        self.kernels = [load(str(path)) for path in paths]
        self.paths = tuple(Path(path) for path in paths)
        by_target = defaultdict(list)
        for kernel in self.kernels:
            for segment in kernel.segments:
                by_target[segment.target].append(segment)
        self._vector_functions = {
            target: segments[0] if len(segments) == 1 else Stack(segments)
            for target, segments in by_target.items()
        }

    def decode(self, target):
        r"""Resolve a body name or ID through Skyfield's kernel decoder.

        :param target: Body name or NAIF identifier.
        :type target: str or int
        :return: Resolved NAIF identifier.
        :rtype: int
        """
        return self.kernels[0].decode(target)

    def __getitem__(self, target):
        r"""Build a vector function for a target across both partitions.

        :param target: Body name or NAIF identifier.
        :type target: str or int
        :return: Skyfield vector function from the solar system barycenter.
        :rtype: skyfield.vectorlib.VectorFunction
        """
        target = self.decode(target)
        vf = self._vector_functions[target]
        if vf.center == 0:
            return vf
        vfs = [vf]
        center = vf.center
        while center in self._vector_functions:
            vf = self._vector_functions[center]
            vfs.append(vf)
            center = vf.center
        return VectorSum(center, target, tuple(reversed(vfs)))

    def close(self):
        r"""Close each underlying BSP partition.

        :return: None
        :rtype: None
        """
        for kernel in self.kernels:
            kernel.close()

    def names(self):
        r"""Merge body name aliases from both DE441 partitions.

        :return: NAIF identifier to aliases mapping.
        :rtype: dict
        """
        merged = {}
        for kernel in self.kernels:
            merged.update(kernel.names())
        return merged


class DE441(DE421):
    """DE441 with the same interface and lifecycle as :class:`DE421`.

    NAIF publishes DE441 as two BSP partitions. Both files are downloaded to
    the user cache on first use and queried together. Their combined size is
    about 3 GB, so construction downloads a substantial amount of data when
    neither file is already available.
    """

    KERNEL_FILENAMES = ("de441_part-1.bsp", "de441_part-2.bsp")
    DOWNLOAD_BASE_URL = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/"
    )

    def __init__(self, filepath: str | Path | None = None) -> None:
        r"""Find or download both DE441 partitions and open them.

        :param filepath: Optional path in the desired partition directory.
        :type filepath: str or pathlib.Path or None
        :return: None
        :rtype: None
        """
        if filepath is None:
            paths = self._find_existing_ephemeris()
            if paths is None:
                paths = self._download_ephemeris(self._get_default_ephemeris_path())
            self.planets = _CombinedKernel(paths)
        else:
            first_path = Path(filepath).expanduser()
            paths = tuple(first_path.parent / name for name in self.KERNEL_FILENAMES)
            if all(path.is_file() for path in paths):
                self.planets = _CombinedKernel(paths)
            else:
                self.planets = _CombinedKernel(
                    self._download_ephemeris(first_path)
                )

        self.sun = self.planets["sun"]
        self.earth = self.planets["earth"]
        self.ts = load.timescale()
        self._closed = False

    @staticmethod
    def _packaged_ephemeris_paths() -> tuple[Path, Path]:
        r"""Get source-package paths for the two DE441 partitions.

        :return: Paths in ``environment/kernels``.
        :rtype: tuple[pathlib.Path, pathlib.Path]
        """
        kernel_dir = Path(__file__).resolve().parents[1] / "kernels"
        return tuple(kernel_dir / name for name in DE441.KERNEL_FILENAMES)

    @staticmethod
    def _cache_ephemeris_paths() -> tuple[Path, Path]:
        r"""Get per-user cache paths for both DE441 partitions.

        :return: Paths honoring the environment override or platform cache.
        :rtype: tuple[pathlib.Path, pathlib.Path]
        """
        override = os.environ.get("ADCS_DE441_EPHEMERIS_PATH")
        if override:
            first = Path(override).expanduser()
            return tuple(first.parent / name for name in DE441.KERNEL_FILENAMES)
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            base = Path(xdg)
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        elif sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            base = Path.home() / ".cache"
        directory = base / "generalized_adcs"
        return tuple(directory / name for name in DE441.KERNEL_FILENAMES)

    @classmethod
    def _find_existing_ephemeris(cls) -> tuple[Path, Path] | None:
        r"""Find a complete pair of locally available DE441 partitions.

        :return: Both paths, or ``None`` if a complete pair is unavailable.
        :rtype: tuple[pathlib.Path, pathlib.Path] or None
        """
        for paths in (cls._cache_ephemeris_paths(), cls._packaged_ephemeris_paths()):
            if all(path.is_file() for path in paths):
                return paths
        return None

    def _get_default_ephemeris_path(self) -> Path:
        r"""Create the cache directory and return the first partition path.

        :return: Destination for ``de441_part-1.bsp``.
        :rtype: pathlib.Path
        """
        paths = self._cache_ephemeris_paths()
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        return paths[0]

    def _download_ephemeris(self, first_path: Path) -> tuple[Path, Path]:
        r"""Download missing DE441 partitions from the JPL NAIF archive.

        :param first_path: Path whose parent holds both partitions.
        :type first_path: pathlib.Path
        :return: Paths to the downloaded or existing partitions.
        :rtype: tuple[pathlib.Path, pathlib.Path]
        :raises RuntimeError: If either partition cannot be downloaded.
        """
        paths = tuple(Path(first_path).parent / name for name in self.KERNEL_FILENAMES)
        try:
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            for filename, destination in zip(self.KERNEL_FILENAMES, paths):
                if destination.is_file():
                    continue
                temporary_path = destination.with_name(destination.name + ".part")
                try:
                    download_with_progress(
                        self.DOWNLOAD_BASE_URL + filename,
                        temporary_path,
                        filename,
                    )
                    temporary_path.replace(destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
            return paths
        except Exception as exc:
            raise RuntimeError(f"❌ Failed to download DE441 ephemeris: {exc}") from exc


__all__ = ["DE441"]
