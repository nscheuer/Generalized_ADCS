import numpy as np
import pytest

from testing.test_estimators._srukf_regression_helpers import nees_metrics, run_srukf_regression_sequence


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nees_result():
    return nees_metrics(run_srukf_regression_sequence(alpha=1.0))


def test_srukf_rate_block_consistency_regression_guard(nees_result):
    _, rate_inside = nees_result
    assert rate_inside > 0.70, f"rate-block 3-sigma consistency regressed: {rate_inside:.0%}"


def test_srukf_attitude_nees_sigma_spread_fix_enforced(nees_result):
    attitude_nees, _ = nees_result
    mean_nees = float(attitude_nees.mean())
    assert mean_nees < 1.0, (
        f"attitude NEES {mean_nees:.3f} >= 1.0 -> sigma-spread or covariance "
        f"calibration regressed."
    )


def test_srukf_attitude_fully_consistent(nees_result):
    attitude_nees, _ = nees_result
    assert float(attitude_nees.mean()) < 0.5, (
        f"attitude NEES {float(attitude_nees.mean()):.1f} not within ~dof+margin; "
        f"attitude process noise is still under-modeled."
    )
