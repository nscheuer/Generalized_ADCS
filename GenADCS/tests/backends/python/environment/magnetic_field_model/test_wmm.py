import pytest

from GenADCS.backends.python.environment.magnetic_field_model.wmm import WMM


def test_wmm_placeholder_reports_unimplemented_model():
    with pytest.raises(NotImplementedError, match="Choose a magnetic field model"):
        WMM().field((6_778_137.0, 0.0, 0.0), 2_458_849.5)
