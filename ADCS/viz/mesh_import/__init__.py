"""Spacecraft model import for the 3-D viewer (glTF preferred, STL accepted).

Run ``python -m ADCS.viz.mesh_import --help``. Needs trimesh and
fast-simplification (``pip install "generalized-adcs[viz-mesh]"``).
"""

from .importer import PRESET_ROTATIONS, UNIT_SCALE, ImportResult, import_spacecraft

__all__ = ["PRESET_ROTATIONS", "UNIT_SCALE", "ImportResult", "import_spacecraft"]
