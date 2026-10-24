"""Download helpers for large ephemeris kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

from tqdm.auto import tqdm


def kernel_path(filename: str) -> Path:
    """Use bundled data when present; cache optional downloads per user."""
    bundled = Path(__file__).resolve().parents[1] / "kernels" / filename
    if bundled.is_file():
        return bundled
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".cache"
    return base / "generalized_adcs" / filename


def download_with_progress(url: str, destination: Path, description: str) -> None:
    r"""Download a kernel while showing byte progress through tqdm.

    :param url: Source URL.
    :type url: str
    :param destination: Local destination path.
    :type destination: pathlib.Path
    :param description: Short label displayed beside the progress bar.
    :type description: str
    :return: None
    :rtype: None
    """
    destination = Path(destination)
    downloaded = 0
    with tqdm(
        total=None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=description,
        leave=True,
    ) as progress:

        def reporthook(block_number: int, block_size: int, total_size: int) -> None:
            nonlocal downloaded
            if total_size > 0 and progress.total is None:
                progress.total = total_size
                progress.refresh()
            current = block_number * block_size
            progress.update(max(0, current - downloaded))
            downloaded = current

        urlretrieve(url, destination, reporthook=reporthook)
