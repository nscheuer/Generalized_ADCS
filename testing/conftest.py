"""Pytest configuration shared across the test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--local",
        action="store_true",
        default=False,
        help=(
            "run local environment integration tests that may use large cached "
            "kernels or check remote data URLs"
        ),
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "local: environment integration/comparison test requiring local kernels or network access",
    )
    if not config.getoption("--local"):
        # Many dynamics tests need planetary positions, but not specifically
        # DE421. Use the bundled offline kernel for standard runs so a clean
        # checkout never downloads the 16 MB DE421 file from inside a test.
        os.environ["ADCS_EPHEMERIS_PATH"] = str(
            REPO_ROOT / "ADCS" / "environment" / "kernels" / "de442s.bsp"
        )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--local"):
        return
    import pytest

    skip_local = pytest.mark.skip(reason="requires pytest --local")
    for item in items:
        if "local" in item.keywords:
            item.add_marker(skip_local)


def pytest_unconfigure(config) -> None:
    xmlpath = getattr(config.option, "xmlpath", None)
    if not xmlpath:
        return

    report_path = Path(xmlpath)
    if not report_path.exists():
        return

    tree = ET.parse(report_path)
    ET.indent(tree, space="  ")
    tree.write(report_path, encoding="utf-8", xml_declaration=True)
