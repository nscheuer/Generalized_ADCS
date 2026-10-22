"""Open the standalone Earth and timeline Panda3D viewer.

Run with: .venv/bin/python debug/debug_helpers/debug_animations/debug_orbit_panda3d.py
"""

from ADCS.helpers.animations import animate_orbit_panda3d


def main() -> None:
    animate_orbit_panda3d()


if __name__ == "__main__":
    main()
