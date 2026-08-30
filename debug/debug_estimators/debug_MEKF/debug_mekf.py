"""Run a focused matrix audit of the multiplicative EKF."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEBUG_HELPERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(DEBUG_HELPERS))

from ADCS.estimators.attitude_estimators import MEKF
from kalman_matrix_debug import run_matrix_debug


if __name__ == "__main__":
    run_matrix_debug(MEKF)
