"""
Conftest for testing: mock compiled extensions that are not available
in the test environment (trajectory_planner, saltro_py).
"""
import sys
from unittest.mock import MagicMock

# Mock compiled C/C++ extension modules so that importing ADCS
# doesn't fail when they are not installed.
for mod in [
    "trajectory_planner",
    "trajectory_planner.build",
    "trajectory_planner.build.tplaunch",
    "trajectory_planner.build.pysat",
    "saltro_py",
    "choldate",
]:
    sys.modules.setdefault(mod, MagicMock())
