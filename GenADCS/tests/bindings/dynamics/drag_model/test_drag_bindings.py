from types import SimpleNamespace

import numpy as np
import pytest


def test_native_drag_functions_accept_numpy_inputs_and_return_vectors(dynamics_backend):
    velocity = np.array([0.0, 0.0, 7500.0], dtype=np.float32)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    arms = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    areas = np.array([2.0], dtype=np.float32)
    coefficients = np.array([2.2], dtype=np.float32)

    force = dynamics_backend.projected_area_drag_force(
        velocity, quaternion, 1e-11, normals, areas, coefficients
    )
    torque = dynamics_backend.projected_area_drag_torque(
        velocity, quaternion, 1e-11, normals, arms, areas, coefficients
    )

    assert isinstance(force, np.ndarray)
    assert isinstance(torque, np.ndarray)
    assert force.shape == torque.shape == (3,)
    assert force.dtype == torque.dtype == np.float64
    assert force[2] < 0.0
    assert np.isfinite(torque).all()


@pytest.mark.parametrize(
    "function,args",
    [
        (
            "projected_area_drag_force",
            (
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 0.0],
                1.0,
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [2.0],
                [2.2],
            ),
        ),
        (
            "projected_area_drag_torque",
            (
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 0.0],
                1.0,
                [[0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [2.0],
                [2.2],
            ),
        ),
    ],
)
def test_native_drag_bindings_reject_mismatched_panel_counts(
    dynamics_backend, function, args
):
    with pytest.raises(ValueError, match="panel arrays must have matching row counts"):
        getattr(dynamics_backend, function)(*args)


def test_drag_model_binding_uses_environment_and_panel_data(dynamics_backend):
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
    model = dynamics_backend.ProjectedAreaDragForceModel(geometry)
    environment = SimpleNamespace(
        atmosphere=SimpleNamespace(
            density=lambda *_: 1.0,
            velocity=lambda *_: np.zeros(3),
        )
    )

    result = model.evaluate(
        np.zeros(3),
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, 0.0],
        0.0,
        environment,
    )
    assert result.shape == (3,)
    assert result[2] < 0.0
