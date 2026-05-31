"""Make the actuator test package deterministic.

This package contains many unseeded stochastic Kolmogorov-Smirnov tests
(``test_*_KS``) whose pass/fail depends on the global ``numpy.random`` state
and therefore on test-collection order. They were observed to fail
intermittently only under full-suite RNG bleed (e.g.
``test_RW_torque_noise_KS``), which is false-confidence flakiness, not a code
defect. Seeding ``numpy.random`` before every test in this package makes the
suite deterministic and order-independent without modifying the individual
tests. KS sample sizes here are large, so a fixed valid PRNG stream still
exercises the distributions correctly.
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_numpy_rng():
    np.random.seed(0xADC50)
    yield
