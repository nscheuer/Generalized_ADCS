from ADCS.satellite_factory import (
    create_3_3_beavercube2_cubesat,
    create_beavercube1_cubesat,
    create_beavercube2_cubesat,
    create_brite_austria,
    create_estcube1_cubesat,
    create_lightsail2,
    create_moveii_cubesat,
    create_rax1_cubesat,
    create_rax2_cubesat,
)
from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GG_Disturbance, SRP_Disturbance


def _hardware_with_biases(sat):
    return list(getattr(sat, "sensors", ())) + list(getattr(sat, "actuators", ()))


def test_all_satellite_factories_include_environment_disturbances_by_default():
    factories = [
        create_beavercube1_cubesat,
        create_beavercube2_cubesat,
        create_3_3_beavercube2_cubesat,
        create_brite_austria,
        create_lightsail2,
        create_estcube1_cubesat,
        create_moveii_cubesat,
        create_rax1_cubesat,
        create_rax2_cubesat,
    ]

    for factory in factories:
        sat = factory()
        assert any(isinstance(dist, GG_Disturbance) for dist in sat.disturbances), factory.__name__
        assert any(isinstance(dist, Drag_Disturbance) for dist in sat.disturbances), factory.__name__
        assert any(isinstance(dist, SRP_Disturbance) for dist in sat.disturbances), factory.__name__


def test_beavercube2_factory_can_disable_hardware_biases():
    sat = create_beavercube2_cubesat(include_biases=False)
    est_sat = create_beavercube2_cubesat(estimated=True, include_biases=False)
    sat_3_3 = create_3_3_beavercube2_cubesat(include_biases=False)

    for hardware in _hardware_with_biases(sat) + _hardware_with_biases(sat_3_3):
        assert not hardware.bias
        assert not hardware.estimate_bias

    for hardware in _hardware_with_biases(est_sat):
        assert not hardware.bias
        assert not hardware.estimate_bias

    assert est_sat.act_bias_len == 0
    assert est_sat.att_sens_bias_len == 0
