from types import SimpleNamespace

import numpy as np
import pytest

from GenADCS.backends.python.dynamics.drag_model import (
    ProjectedAreaDragForceModel,
    ProjectedAreaDragTorqueModel,
    projected_area_drag_force,
    projected_area_drag_torque,
)


def test_drag_force_opposes_flow_and_scales_with_density():
    normals = np.array([[0.0, 0.0, 1.0]])
    args = (
        np.array([0.0, 0.0, 7500.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        1e-11,
        normals,
        np.array([2.0]),
        np.array([2.2]),
    )
    force = projected_area_drag_force(*args)
    assert force[2] < 0 and np.linalg.norm(force) > 0
    np.testing.assert_allclose(
        projected_area_drag_force(*args[:2], 2e-11, *args[3:]), 2 * force
    )
    np.testing.assert_allclose(
        projected_area_drag_force(*args[:2], 0.0, *args[3:]), np.zeros(3)
    )


def test_drag_torque_is_moment_of_panel_force():
    normal = np.array([[0.0, 0.0, 1.0]])
    arm = np.array([[1.0, 0.0, 0.0]])
    args = (
        np.array([0.0, 0.0, 100.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        1.0,
        normal,
        arm,
        np.array([2.0]),
        np.array([2.0]),
    )
    force = projected_area_drag_force(
        args[0], args[1], args[2], normal, args[5], args[6]
    )
    torque = projected_area_drag_torque(*args)
    np.testing.assert_allclose(torque, np.cross(arm[0], force))


def test_drag_models_require_atmosphere_and_evaluate_with_panel_geometry():
    panel = SimpleNamespace(
        orientation=np.zeros(3),
        position=np.array([1.0, 0.0, 0.0]),
        dimensions=np.array([1.0, 1.0]),
        eta_s=0.0,
        eta_d=0.0,
        eta_a=1.0,
        cd=2.2,
    )
    geometry = SimpleNamespace(panels=[panel], center_of_mass=np.zeros(3))
    model = ProjectedAreaDragForceModel(geometry)
    with pytest.raises(ValueError, match="requires environment.atmosphere"):
        model.evaluate(
            np.zeros(3),
            np.array([0.0, 0.0, 10.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
            SimpleNamespace(atmosphere=None),
        )
    env = SimpleNamespace(
        atmosphere=SimpleNamespace(
            density=lambda *_: 1.0, velocity=lambda *_: np.zeros(3)
        )
    )
    assert (
        model.evaluate(
            np.zeros(3),
            np.array([0.0, 0.0, 10.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
            env,
        )[2]
        < 0
    )
    assert np.isfinite(
        ProjectedAreaDragTorqueModel(geometry).evaluate(
            np.zeros(3),
            np.array([0.0, 0.0, 10.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
            env,
        )
    ).all()
