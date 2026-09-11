"""End-to-end AugmentedMEKF test estimating a residual magnetic dipole."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_disturbance_scenario import run
import ADCS


if __name__ == "__main__":
    run(ADCS.AugmentedMEKF, "Augmented MEKF: Dipole Parameter", direct_torque=False)
