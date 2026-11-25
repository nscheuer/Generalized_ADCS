"""Numerical and field parity for SatellitePanel across backends."""


from typing import Any


def test_panel_fields_match_pairwise(geometry_backends: Any) -> None:
    results: dict[str, Any] = {
        name: api.SatellitePanel(
            position=(1.0, -2.0, 3.5),
            orientation=(0.0, 0.6, 0.8),
            dimensions=(2.5, 1.25),
            eta_s=0.15,
            eta_d=0.25,
            eta_a=0.60,
            cd=2.3,
        )
        for name, api in geometry_backends.items()
    }
    reference = results["python"]
    fields = ("position", "orientation", "dimensions", "eta_s", "eta_d", "eta_a", "cd")
    for backend, panel in results.items():
        for field in fields:
            assert getattr(panel, field) == getattr(reference, field), (
                f"{backend}.{field} differs"
            )


def test_panel_field_updates_match_pairwise(geometry_backends: Any) -> None:
    results: dict[str, Any] = {}
    for name, api in geometry_backends.items():
        panel = api.SatellitePanel((0, 0, 0), (1, 0, 0), (1, 1))
        panel.position = (4.0, 5.0, 6.0)
        panel.eta_s, panel.eta_d, panel.eta_a, panel.cd = 0.1, 0.2, 0.7, 2.4
        results[name] = panel
    reference = results["python"]
    for backend, panel in results.items():
        assert panel.position == reference.position, f"{backend}.position differs"
        assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (
            reference.eta_s,
            reference.eta_d,
            reference.eta_a,
            reference.cd,
        ), f"{backend} coefficients differ"
