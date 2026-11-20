"""Parity checks for the C++ satellite geometry translation."""
from __future__ import annotations

import numpy as np

from ADCS_Omni.ADCS_C.hardware.satellite_geometry import SatelliteGeometry as CGeometry
from ADCS_Omni.ADCS_C.hardware.satellite_geometry import SatellitePanel as CPanel
from ADCS_Omni.ADCS_Python.hardware.satellite_geometry import SatelliteGeometry, SatellitePanel


def main() -> None:
    inertia = np.diag([2.0, 3.0, 4.0])
    py_panel = SatellitePanel((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 1.0))
    c_panel = CPanel((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 1.0))
    for field in ("position", "orientation", "dimensions", "eta_s", "eta_d", "eta_a", "cd"):
        if getattr(c_panel, field) != getattr(py_panel, field):
            raise AssertionError(f"SatellitePanel.{field} differs")

    py_panels = [py_panel]
    c_panels = [c_panel]
    py_geometry = SatelliteGeometry(12.5, inertia, panels=py_panels)
    c_geometry = CGeometry(12.5, inertia, panels=c_panels)
    if c_geometry.mass != py_geometry.mass:
        raise AssertionError("SatelliteGeometry.mass differs")
    if not np.array_equal(c_geometry.inertia, py_geometry.inertia):
        raise AssertionError("SatelliteGeometry.inertia differs")
    if c_geometry.panels != c_panels or c_geometry.center_of_mass != py_geometry.center_of_mass:
        raise AssertionError("SatelliteGeometry panels or center of mass differ")
    c_panels.append(CPanel((0, 0, 0), (0, 1, 0), (1, 1)))
    if len(c_geometry.panels) != 2:
        raise AssertionError("SatelliteGeometry did not preserve the supplied mutable panel list")

    default_py = SatelliteGeometry(1.0, np.eye(3))
    default_c = CGeometry(1.0, np.eye(3))
    if default_c.center_of_mass != default_py.center_of_mass or default_c.panels != default_py.panels:
        raise AssertionError("SatelliteGeometry defaults differ")
    print("C++ satellite geometry translation matches the Python reference for tested behavior.")


if __name__ == "__main__":
    main()
