r"""
Phase 0 hardening for multi-satellite simulation:

- ``Satellite``/``EstimatedSatellite`` no longer share mutable default lists.
- ``Bias``/``Noise``/``AnisotropicNoise`` draw from an optional per-instance
  ``numpy.random.Generator`` (falling back to global ``np.random`` when unset).
- ``Satellite.set_rng`` / ``Satellite.seed`` distribute a per-satellite generator
  to every stochastic hardware model, so many satellites in one process are
  independently reproducible.
"""

import numpy as np

from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.errors.anisotropicnoise import AnisotropicNoise
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants


# --------------------------------------------------------------------------- #
# Mutable default arguments must not be shared across instances
# --------------------------------------------------------------------------- #
def test_satellite_default_lists_are_independent_instances():
    s1 = Satellite()
    s2 = Satellite()
    assert s1.sensors is not s2.sensors
    assert s1.actuators is not s2.actuators
    assert s1.disturbances is not s2.disturbances
    s1.sensors.append("x")
    s1.actuators.append("y")
    assert s2.sensors == []
    assert s2.actuators == []


def test_estimated_satellite_default_lists_are_independent():
    e1 = EstimatedSatellite()
    e2 = EstimatedSatellite()
    assert e1.sensors is not e2.sensors
    e1.disturbances.append("z")
    assert e2.disturbances == []


# --------------------------------------------------------------------------- #
# Per-instance RNG: reproducible and independent
# --------------------------------------------------------------------------- #
def test_bias_random_walk_is_seed_reproducible_and_independent():
    def walk(seed):
        b = Bias(bias=np.zeros(1), std_bias=np.ones(1), rng=np.random.default_rng(seed))
        b._update_bias(0.0)  # initializes last_bias_time
        vals = []
        for k in range(1, 6):
            b._update_bias(k * 1e-6)
            vals.append(float(np.atleast_1d(b.bias)[0]))
        return np.array(vals)

    assert np.array_equal(walk(0), walk(0))      # same seed -> identical
    assert not np.allclose(walk(0), walk(1))     # different seed -> independent


def test_noise_sampling_is_seed_reproducible_and_independent():
    def draws(seed):
        n = Noise(noise=np.zeros(2), std_noise=np.ones(2), rng=np.random.default_rng(seed))
        out = []
        for _ in range(5):
            n._update_noise()
            out.append(np.asarray(n.noise, dtype=float).copy())
        return np.array(out)

    assert np.array_equal(draws(2), draws(2))
    assert not np.allclose(draws(2), draws(3))


def test_anisotropic_noise_uses_per_instance_rng():
    n1 = AnisotropicNoise(std_cross=1.0, std_roll=2.0, rng=np.random.default_rng(7))
    n2 = AnisotropicNoise(std_cross=1.0, std_roll=2.0, rng=np.random.default_rng(7))
    n1._update_noise()
    n2._update_noise()
    assert np.array_equal(n1.noise, n2.noise)


def test_rng_none_falls_back_to_global_np_random():
    # Back-compat: with rng unset the draw comes from the global np.random stream.
    n = Noise(noise=np.zeros(1), std_noise=np.ones(1))
    np.random.seed(12345)
    n._update_noise()
    got = float(np.atleast_1d(n.noise)[0])
    np.random.seed(12345)
    exp = float(np.atleast_1d(np.random.normal(loc=0.0, scale=np.ones(1)))[0])
    assert got == exp


# --------------------------------------------------------------------------- #
# Satellite-level RNG distribution
# --------------------------------------------------------------------------- #
def _make_noisy_sat():
    mtqs = [
        MTQ(axis=ax, max_torque=1.0,
            bias=Bias(bias=0.0, std_bias=1e-3),
            noise=Noise(noise=0.0, std_noise=1e-3))
        for ax in MathConstants.unitvecs
    ]
    return Satellite(actuators=mtqs)


def test_set_rng_distributes_to_every_error_model():
    sat = _make_noisy_sat()
    gen = np.random.default_rng(0)
    out = sat.set_rng(gen)
    assert out is sat  # chainable
    for act in sat.actuators:
        assert act.noise.rng is gen
        assert act.bias.rng is gen


def test_seed_makes_satellites_reproducible_and_independent():
    def actuator_noise_sequence(seed):
        sat = _make_noisy_sat()
        sat.seed(seed)
        seq = []
        for act in sat.actuators:
            act.noise._update_noise()
            seq.append(np.asarray(act.noise.noise, dtype=float).copy())
        return np.array(seq)

    assert np.array_equal(actuator_noise_sequence(5), actuator_noise_sequence(5))
    assert not np.allclose(actuator_noise_sequence(5), actuator_noise_sequence(6))


def test_two_seeded_satellites_do_not_share_rng_state():
    a = _make_noisy_sat().seed(10)
    b = _make_noisy_sat().seed(11)
    # Distinct generators on distinct instances.
    assert a.actuators[0].noise.rng is not b.actuators[0].noise.rng
