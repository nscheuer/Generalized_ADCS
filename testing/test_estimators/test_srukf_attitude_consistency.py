import pytest

from testing.test_estimators._srukf_regression_helpers import consistency_percentages, run_srukf_regression_sequence


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def srukf_consistency_result():
    return consistency_percentages(run_srukf_regression_sequence())


def test_srukf_rate_block_consistency_regression_guard(srukf_consistency_result):
    rate_percentages, _ = srukf_consistency_result
    for axis, percentage in enumerate(rate_percentages):
        assert percentage > 0.70, f"rate-block consistency regressed on axis {axis}: {percentage:.0%}"


def test_srukf_attitude_block_consistency(srukf_consistency_result):
    _, attitude_percentages = srukf_consistency_result
    worst = min(attitude_percentages)
    assert worst > 0.95, (
        f"attitude-block 3-sigma consistency too low: per-axis {attitude_percentages} "
        f"(worst {worst:.0%}); attitude covariance consistency regressed."
    )
