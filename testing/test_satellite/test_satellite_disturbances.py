import numpy as np

from ADCS.satellite_hardware.disturbances import (
    Dipole_Disturbance,
    Drag_Disturbance,
    GeometryConfig,
    GeometryFace,
    Prop_Disturbance,
    SRP_Disturbance,
)
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import Satellite

from testing.test_satellite._helpers import make_orbital_state


def make_geometry_faces():
    return [
        GeometryFace(
            area=0.1,
            centroid=np.array([1.0, 0.2, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            eta_s=0.0,
            eta_d=0.5,
            eta_a=0.5,
            CD=2.0,
        ),
        GeometryFace(
            area=0.03,
            centroid=np.array([-0.05, 0.1, 0.3]),
            normal=np.array([0.0, 1.0, 0.0]),
            eta_s=0.1,
            eta_d=0.2,
            eta_a=0.1,
            CD=0.1,
        ),
        GeometryFace(
            area=10.0,
            centroid=np.array([0.25, -0.01, -0.7]),
            normal=np.array([0.0, 0.0, 1.0]),
            eta_s=0.3,
            eta_d=0.1,
            eta_a=0.6,
            CD=0.3,
        ),
    ]


def test_srp_disturbance_loads_geometry_configuration():
    sat = Satellite(disturbances=[SRP_Disturbance(config=GeometryConfig(geometry_faces=make_geometry_faces()))])
    disturbance = sat.disturbances[0]

    assert np.allclose(disturbance.eta_s, [0.0, 0.1, 0.3])
    assert np.allclose(disturbance.eta_d, [0.5, 0.2, 0.1])
    assert np.allclose(disturbance.eta_a, [0.5, 0.1, 0.6])
    assert np.allclose(disturbance.areas, [0.1, 0.03, 10.0])
    assert np.allclose(disturbance.normals[0], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(disturbance.normals[1], np.array([0.0, 1.0, 0.0]))
    assert np.allclose(disturbance.normals[2], np.array([0.0, 0.0, 1.0]))


def test_drag_disturbance_loads_geometry_configuration():
    sat = Satellite(disturbances=[Drag_Disturbance(config=GeometryConfig(geometry_faces=make_geometry_faces()))])
    disturbance = sat.disturbances[0]

    assert np.allclose(disturbance.areas, [0.1, 0.03, 10.0])
    assert np.allclose(disturbance.centroids[0], np.array([1.0, 0.2, 0.0]))
    assert np.allclose(disturbance.centroids[1], np.array([-0.05, 0.1, 0.3]))
    assert np.allclose(disturbance.centroids[2], np.array([0.25, -0.01, -0.7]))
    assert np.allclose(disturbance.CDs, [2.0, 0.1, 0.3])


def test_prop_disturbance_returns_nominal_torque():
    disturbance = Prop_Disturbance(np.array([1.0, 2.0, 4.0]), Noise())
    sat = Satellite(disturbances=[disturbance])
    x = np.concatenate([np.array([0.2, -0.1, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])])

    assert np.allclose(sat.disturbances[0].torque(x=x, os=make_orbital_state()), np.array([1.0, 2.0, 4.0]))


def test_dipole_disturbance_preserves_nominal_configuration():
    disturbance = Dipole_Disturbance(np.array([0.1, -0.1, 0.5]), Noise())
    sat = Satellite(disturbances=[disturbance])

    assert np.allclose(sat.disturbances[0].torque_nominal, np.array([0.1, -0.1, 0.5]))
    assert np.allclose(sat.disturbances[0].noise.std_noise, 0.0)
