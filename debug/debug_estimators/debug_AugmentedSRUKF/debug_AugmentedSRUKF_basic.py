import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from debug.debug_estimators.augmented_ukf_scenarios import run_basic
from ADCS import AugmentedSRUKF

if __name__ == "__main__":
    run_basic(AugmentedSRUKF, "Augmented SRUKF: Basic")
