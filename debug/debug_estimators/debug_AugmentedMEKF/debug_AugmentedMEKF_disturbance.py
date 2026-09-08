"""End-to-end AugmentedMEKF test with direct total-disturbance torque estimation."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_disturbance_scenario import run
import ADCS


if __name__ == "__main__":
    run(ADCS.AugmentedMEKF, "Augmented MEKF: Direct Disturbance Torque", direct_torque=True)
