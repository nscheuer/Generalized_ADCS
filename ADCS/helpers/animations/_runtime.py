"""Host-specific Panda3D configuration, applied before ShowBase starts."""

import sys
from pathlib import Path


def configure_runtime(load_prc_file_data) -> None:
    """Avoid optional Linux controller discovery when no device node exists."""
    if sys.platform.startswith("linux") and not Path("/dev/input").is_dir():
        load_prc_file_data("Generalized ADCS capability detection", "notify-level-device fatal")

    load_prc_file_data("Generalized ADCS viewer", "window-title Generalized ADCS | Orbit Viewer")
    load_prc_file_data("Generalized ADCS viewer", "win-size 1280 800")
    load_prc_file_data("Generalized ADCS viewer", "framebuffer-srgb true")
    load_prc_file_data("Generalized ADCS viewer", "sync-video false")
    load_prc_file_data("Generalized ADCS viewer", "audio-library-name null")
