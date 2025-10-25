import sys
import os

# Add the build directory to sys.path
this_dir = os.path.dirname(__file__)
build_path = os.path.join(this_dir, "trajectory_planner", "build")
sys.path.append(build_path)

import tplaunch
import pysat

print("Successfully imported C++ modules!")
