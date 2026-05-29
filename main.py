import numpy as np

from ADCS.controller.helpers.optional_dependencies import get_trajectory_planner_modules

try:
    tplaunch, pysat = get_trajectory_planner_modules()
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

