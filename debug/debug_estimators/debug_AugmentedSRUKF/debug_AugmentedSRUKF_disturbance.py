import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_disturbance_scenario import run
from ADCS import AugmentedSRUKF

if __name__ == "__main__":
    run(AugmentedSRUKF, "Augmented SRUKF: Direct Disturbance Torque", direct_torque=True)
