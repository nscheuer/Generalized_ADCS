from pathlib import Path

import pytest

from GenADCS.backends.python.environment.magnetic_field_model.magnetic_field_model import read_shc


def test_shc_coefficients_are_cached():
    read_shc.cache_clear()
    path = str(
        Path(__file__).resolve().parents[5]
        / "kernels/environment/magnetic_field_model/igrf14/IGRF14.shc"
    )
    first = read_shc(path, 1, 13)
    second = read_shc(path, 1, 13)

    assert first is second
    assert read_shc.cache_info().hits == 1


@pytest.mark.parametrize(
    "contents,match",
    (
        ("1 2 1 0 0\n2020\n1 0\n", "Unexpected harmonic degrees"),
        ("1 1 2 0 0\n2020\n1 0 1\n", "Invalid SHC epochs"),
        ("1 1 1 0 0\n2020\n1 0\n", "Malformed SHC row"),
    ),
)
def test_shc_reader_rejects_malformed_coefficient_files(tmp_path, contents, match):
    path = tmp_path / "bad.shc"
    path.write_text(contents, encoding="ascii")
    read_shc.cache_clear()

    with pytest.raises(ValueError, match=match):
        read_shc(str(path), 1, 1)
