import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_disturbance_scenario import run
from ADCS import AugmentedUKF

if __name__ == "__main__":
    run(AugmentedUKF, "Augmented UKF: Direct Disturbance Torque", direct_torque=True)
