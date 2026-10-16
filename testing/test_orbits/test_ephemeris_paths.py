"""Ephemeris file resolution: never write into site-packages.

de421.bsp is 16 MB and is deliberately not shipped in the wheel, so a
pip-installed user downloads it on first use. It previously landed *inside*
site-packages, which:

* hard-failed with a misleading "Failed to download" RuntimeError on read-only
  or system-managed installs, when the real cause was a PermissionError; and
* survived ``pip uninstall`` as a 16 MB orphan, being absent from RECORD.

These tests pin the search order and the write location. They never download
anything.
"""

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from ADCS.orbits.ephemeris import Ephemeris


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("ADCS_DE442_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("ADCS_DE441_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_download_target_is_not_inside_site_packages():
    """The regression that motivated all of this."""
    target = Ephemeris._cache_ephemeris_path()
    parts = [p.lower() for p in target.parts]
    assert "site-packages" not in parts, f"would write into site-packages: {target}"
    assert "dist-packages" not in parts, f"would write into dist-packages: {target}"


def test_cache_path_is_absolute_and_named():
    target = Ephemeris._cache_ephemeris_path()
    assert target.is_absolute()
    assert target.name == "de421.bsp"
    assert "generalized_adcs" in target.parts


def test_env_override_wins(monkeypatch, tmp_path):
    custom = tmp_path / "somewhere" / "my_de421.bsp"
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))
    assert Ephemeris._cache_ephemeris_path() == custom


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", "~/eph/de421.bsp")
    resolved = Ephemeris._cache_ephemeris_path()
    assert "~" not in str(resolved)
    assert resolved.is_absolute()


def test_xdg_cache_home_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = Ephemeris._cache_ephemeris_path()
    assert target == tmp_path / "generalized_adcs" / "de421.bsp"


def test_platform_default_without_xdg(monkeypatch):
    target = Ephemeris._cache_ephemeris_path()
    home = Path.home()
    if sys.platform == "darwin":
        assert target.is_relative_to(home / "Library" / "Caches")
    elif sys.platform == "win32":
        assert target.is_absolute()
    else:
        assert target.is_relative_to(home / ".cache")


def test_packaged_path_points_at_the_package(monkeypatch):
    packaged = Ephemeris._packaged_ephemeris_path()
    assert packaged.name == "de421.bsp"
    assert packaged.parent.name == "kernels"
    # must sit under the ADCS package, not the repo root
    assert packaged.parent.parent.parent.name == "ADCS"


def test_search_order_prefers_the_cache_over_the_packaged_copy(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "de421.bsp"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"not a real ephemeris")
    packaged = tmp_path / "pkg" / "de421.bsp"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"also not real")

    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: cache))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: packaged))
    assert Ephemeris._find_existing_ephemeris() == cache


def test_falls_back_to_the_packaged_copy(monkeypatch, tmp_path):
    missing = tmp_path / "cache" / "de421.bsp"
    packaged = tmp_path / "pkg" / "de421.bsp"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"stand-in")

    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: missing))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: packaged))
    assert Ephemeris._find_existing_ephemeris() == packaged


def test_returns_none_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path",
                        staticmethod(lambda: tmp_path / "a" / "de421.bsp"))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path",
                        staticmethod(lambda: tmp_path / "b" / "de421.bsp"))
    assert Ephemeris._find_existing_ephemeris() is None


def test_default_download_path_creates_only_its_own_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = Ephemeris()._get_default_ephemeris_path.__func__(Ephemeris)
    assert target.parent.is_dir(), "the cache directory should be created"
    assert target.parent == tmp_path / "generalized_adcs"


def test_resolution_does_not_touch_the_package_directory(monkeypatch, tmp_path):
    """Resolving a path must not mkdir inside the installed package."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pkg_env_dir = Ephemeris._packaged_ephemeris_path().parent
    existed_before = pkg_env_dir.exists()
    Ephemeris._cache_ephemeris_path()
    Ephemeris._find_existing_ephemeris()
    assert pkg_env_dir.exists() == existed_before, (
        "path resolution must not create directories inside the package")


def test_de421_loads_existing_kernel_without_downloading(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE421
    import importlib

    module = importlib.import_module("ADCS.environment.ephemeris.de421")
    existing = tmp_path / "de421.bsp"
    existing.write_bytes(b"stand-in")
    calls = []

    class FakePlanets:
        def __getitem__(self, name):
            return name

        def close(self):
            pass

    class FakeLoad:
        def __call__(self, path):
            calls.append(path)
            return FakePlanets()

        def timescale(self):
            return object()

    monkeypatch.setattr(DE421, "_find_existing_ephemeris", classmethod(lambda cls: existing))
    monkeypatch.setattr(module, "load", FakeLoad())
    monkeypatch.setattr(DE421, "_download_ephemeris", lambda self, path: pytest.fail("unexpected download"))

    with DE421() as model:
        assert model.sun == "sun"
    assert calls == [str(existing)]


def test_de421_downloads_when_no_kernel_exists(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE421
    import importlib

    module = importlib.import_module("ADCS.environment.ephemeris.de421")
    target = tmp_path / "cache" / "de421.bsp"
    calls = []

    class FakePlanets:
        path = target

        def __getitem__(self, name):
            return name

        def close(self):
            pass

    class FakeLoader:
        def __init__(self, directory):
            calls.append(("directory", directory))

        def build_url(self, filename):
            calls.append(("url", filename))
            return "https://example.test/" + filename

    class FakeLoad:
        def __call__(self, path):
            return FakePlanets()

        def timescale(self):
            return object()

    def fake_download(url, destination, description):
        calls.append(("download", url, description))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"kernel")

    monkeypatch.setattr(DE421, "_find_existing_ephemeris", classmethod(lambda cls: None))
    monkeypatch.setattr(DE421, "_get_default_ephemeris_path", lambda self: target)
    monkeypatch.setattr(module, "Loader", FakeLoader)
    monkeypatch.setattr(module, "load", FakeLoad())
    monkeypatch.setattr(module, "download_with_progress", fake_download)

    with DE421() as model:
        assert model.earth == "earth"
    assert calls == [
        ("directory", str(target.parent)),
        ("url", "de421.bsp"),
        ("download", "https://example.test/de421.bsp", "de421.bsp"),
    ]


def test_de421_downloads_to_explicit_missing_path(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE421
    import importlib

    module = importlib.import_module("ADCS.environment.ephemeris.de421")
    target = tmp_path / "kernels" / "custom.bsp"
    calls = []

    class FakePlanets:
        def __init__(self, path):
            self.path = path

        def __getitem__(self, name):
            return name

        def close(self):
            pass

    class FakeLoader:
        def __init__(self, directory):
            pass

        def build_url(self, filename):
            return "https://example.test/" + filename

    class FakeLoad:
        def __call__(self, path):
            calls.append(("load", path))
            return FakePlanets(Path(path))

        def timescale(self):
            return object()

    def fake_download(url, destination, description):
        calls.append(("fetch", description))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"downloaded kernel")

    monkeypatch.setattr(module, "Loader", FakeLoader)
    monkeypatch.setattr(module, "load", FakeLoad())
    monkeypatch.setattr(module, "download_with_progress", fake_download)

    with DE421(filepath=target) as model:
        assert model.sun == "sun"
    assert target.read_bytes() == b"downloaded kernel"
    assert calls == [("fetch", "de421.bsp"), ("load", str(target))]


def test_de442_downloads_from_naif_on_first_use(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE442
    import importlib

    module = importlib.import_module("ADCS.environment.ephemeris.de442")
    target = tmp_path / "cache" / "de442.bsp"
    requested = []

    class FakePlanets:
        def __getitem__(self, name):
            return name

        def close(self):
            pass

    class FakeLoad:
        def __call__(self, path):
            requested.append(("load", path))
            return FakePlanets()

        def timescale(self):
            return object()

    def fake_download(url, destination, description):
        requested.append(("download", url))
        Path(destination).write_bytes(b"downloaded DE442 kernel")
        requested.append(("description", description))

    monkeypatch.setattr(DE442, "_find_existing_ephemeris", classmethod(lambda cls: None))
    monkeypatch.setattr(DE442, "_get_default_ephemeris_path", lambda self: target)
    monkeypatch.setattr(module, "download_with_progress", fake_download)
    monkeypatch.setattr(module, "load", FakeLoad())

    with DE442() as model:
        assert model.earth == "earth"
        assert model.sun == "sun"

    assert target.read_bytes() == b"downloaded DE442 kernel"
    assert requested == [
        ("download", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp"),
        ("description", "de442.bsp"),
        ("load", str(target)),
    ]


def test_de442_uses_a_separate_cache_filename(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE442

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert DE442._cache_ephemeris_path() == tmp_path / "generalized_adcs" / "de442.bsp"


@pytest.mark.parametrize("url", [
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-1.bsp",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-2.bsp",
    "https://ftp.imcce.fr/pub/ephem/planets/vsop2013/ephemerides/VSOP2013.p2000",
])
def test_ephemeris_data_urls_are_reachable_without_downloading(url):
    """Use HTTP HEAD so large ephemeris data files are never transferred."""
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=15) as response:
        assert response.status == 200
