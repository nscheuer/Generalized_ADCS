"""Viewer runtime files stay outside a read-only installation."""

from pathlib import Path

from ADCS.helpers.animations.godot import binary_manager, launcher
from ADCS.helpers.animations.godot.assets.fetch import cache_dir


def test_godot_binary_and_assets_use_user_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(binary_manager, "is_wsl", lambda: False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("ADCS_VIZ_CACHE", raising=False)
    root = tmp_path / "cache" / "generalized_adcs"
    assert binary_manager._executable("linux-x86_64").is_relative_to(root / "godot")
    assert cache_dir() == root / "viewer_assets"


def test_wsl_assets_use_windows_visible_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ADCS_VIZ_CACHE", raising=False)
    monkeypatch.setattr(binary_manager, "is_wsl", lambda: True)
    monkeypatch.setattr(binary_manager, "_windows_local_appdata", lambda: tmp_path / "windows_local")
    assert cache_dir() == tmp_path / "windows_local" / "generalized_adcs" / "viewer_assets"


def test_wsl_windows_binary_is_copied_to_windows_local_cache(monkeypatch, tmp_path):
    source = tmp_path / "wsl_cache" / "Godot.exe"
    source.parent.mkdir()
    source.write_bytes(b"godot stand-in")
    monkeypatch.setattr(binary_manager, "_windows_local_appdata", lambda: tmp_path / "windows_local")
    result = binary_manager._wsl_windows_executable(source)
    assert result == tmp_path / "windows_local" / "generalized_adcs" / "viewer" / binary_manager._GODOT_VERSION / source.name
    assert result.read_bytes() == source.read_bytes()


def test_bundled_project_is_copied_for_own_godot(monkeypatch, tmp_path):
    source = tmp_path / "read_only_package" / "viewer"
    source.mkdir(parents=True)
    (source / "project.godot").write_text("[application]")
    monkeypatch.setattr(launcher, "_REPO_VIEWER", source)
    monkeypatch.setattr(launcher, "is_wsl", lambda: False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = launcher.viewer_project(Path("/usr/bin/godot"))

    assert result.parent == tmp_path / "cache" / "generalized_adcs" / "viewer_project"
    assert (result / "project.godot").read_text() == "[application]"
    assert not (source / ".godot").exists()
    (source / "project.godot").write_text("[updated application]")
    updated = launcher.viewer_project(Path("/usr/bin/godot"))
    assert updated != result
    assert (updated / "project.godot").read_text() == "[updated application]"


def test_wsl_windows_godot_project_uses_windows_local_cache(monkeypatch, tmp_path):
    source = tmp_path / "read_only_package" / "viewer"
    source.mkdir(parents=True)
    (source / "project.godot").write_text("[application]")
    monkeypatch.setattr(launcher, "_REPO_VIEWER", source)
    monkeypatch.setattr(launcher, "is_wsl", lambda: True)
    monkeypatch.setattr(launcher, "_windows_local_appdata", lambda: tmp_path / "windows_local")

    result = launcher.viewer_project(Path("/mnt/c/Godot.exe"))

    assert result.parent == tmp_path / "windows_local" / "generalized_adcs" / "viewer_project"
    assert (result / "project.godot").is_file()
