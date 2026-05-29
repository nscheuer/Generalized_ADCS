"""Pytest configuration shared across the test suite."""

from __future__ import annotations

import sys
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
