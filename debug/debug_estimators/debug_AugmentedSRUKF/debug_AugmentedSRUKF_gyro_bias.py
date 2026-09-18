import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_ukf_scenarios import run_gyro_bias
from ADCS import AugmentedSRUKF

if __name__ == "__main__":
    run_gyro_bias(AugmentedSRUKF, "Augmented SRUKF: Gyro Bias")
