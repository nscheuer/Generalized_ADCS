"""Run the new MEKF with a closed-loop attitude controller."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from debug.debug_estimators.controlled_attitude_scenario import run
from ADCS import MEKF


if __name__ == "__main__":
    run(MEKF, "MEKF")

