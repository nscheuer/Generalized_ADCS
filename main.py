import sys
import os

this_dir = os.path.dirname(__file__)
build_path = os.path.join(this_dir, "trajectory_planner", "build")
sys.path.append(build_path)

import numpy as np
import tplaunch
import pysat

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.orbits.ephemeris import Ephemeris

if __name__ == "__main__":
    mass = 1.0
    COM = np.array([0, 0, 0])
    J_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sat = Satellite(mass, COM, J_0)

    ephem = Ephemeris()
    print(type(ephem.planets['earth']))

