import sys
import os

this_dir = os.path.dirname(__file__)
# trajectory_planner now lives in the OldPlanner submodule
# (https://github.com/patrickmckeen/OldPlanner), built into OldPlanner/build/.
build_path = os.path.join(this_dir, "OldPlanner", "build")
sys.path.append(build_path)

import numpy as np
try:
    import tplaunch
    import pysat
except ImportError as exc:
    raise ImportError(
        "Optional add-on trajectory_planner (OldPlanner) is not available. "
        "Initialise the OldPlanner submodule and build into OldPlanner/build/ "
        "first (see docs/Install_Trajectory_Planner.md)."
    ) from exc

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.orbits.ephemeris import Ephemeris

if __name__ == "__main__":
    mass = 1.0
    COM = np.array([0, 0, 0])
    J_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sat = Satellite(mass, COM, J_0)

    ephem = Ephemeris()
    print(type(tplaunch.Planner))

