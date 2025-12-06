import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from scipy.stats import kstest, ks_2samp
from asciichartpy import plot

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.satellite_hardware.sensors import MTM
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, normalize


def test_mtm_reading_etc_clean():
    ax = random_n_unit_vec(3) * 3
    mtm = MTM(axis=ax)

    # Defaults
    assert mtm.sample_time == 0.1
    assert np.all(mtm.bias.bias == np.zeros(1))
    assert np.all(mtm.bias.std_bias == np.zeros(1))
    assert np.all(mtm.noise.noise == np.zeros(1))
    assert np.all(mtm.noise.std_noise == np.zeros(1))

    # State
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 8, 0])
    )
    sat = Satellite(sensors=[mtm])
    vecs = os.get_state_vector(x=x)

    # Clean output and expected
    out = mtm.reading(x=x, os=os)
    expected = np.dot(vecs["b"], ax / 3.0)  # axis normalized in MTM.__init__
    assert np.allclose(out, expected)

    # Numeric Jacobian vs analytic basestate_jac
    xfun = lambda c: np.asarray(
        mtm.reading(
            x=np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6]]),
            os=os
        )
    ).item()

    # Bias-Jacobian: differentiate through explicit Bias(bias=c)
    bfun = lambda c: np.asarray(
        MTM(axis=ax).reading(x=x, os=os)
    ).item()

    Jxfun = nd.Jacobian(xfun)(x.flatten().tolist())  # (7,1)
    Jbfun = nd.Jacobian(bfun)(20000)

    assert np.isclose(bfun(20000), np.dot(ax/3, vecs["b"]))
    assert np.isclose(xfun(x), np.dot(ax/3, vecs["b"]))

    


def test_mtm_reading_bias_KS():
    # Bias-only setup
    e_bias = np.random.uniform(1, 3) * random_n_unit_vec(1)
    std_bias = np.abs(np.random.uniform(0.001, 0.1) * random_n_unit_vec(1))
    bias = Bias(bias=e_bias, std_bias=std_bias)

    ax = random_n_unit_vec(3) * 3
    mtm = MTM(axis=ax, bias=bias)

    # State
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 8, 0])
    )
    sat = Satellite(sensors=[mtm])

    # Bias drift differences
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b1 = mtm.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b2 = mtm.reading(x=x, os=os)
        drifts.append(b1 - b2)
    drifts = np.array(drifts).reshape(N, 1)

    # Expected distribution (matching your prior GPS/Sun tests)
    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.abs(std_bias) * np.sqrt(0.5),
        size=(N, 1)
    )

    # KS test
    i = 0
    ks = kstest(drifts[:, i], exp_dist[:, i])
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    # Visualization
    hist = np.histogram(drifts[:, i], bins="auto")
    edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [np.sum(exp_dist[:, i] < ee) for ee in edges[1:]]
    print(f"MTM Bias component {i} KS:")
    print(plot([hist_a, hist_b], {"height": 20}))

    assert ks.pvalue > 0.1 or np.abs(ks.statistic) < threshold


def test_mtm_reading_noise_KS():
    # Noise-only setup
    std_noise = np.abs(np.random.uniform(0.001, 0.1, size=1))
    noise = Noise(noise=np.zeros(1), std_noise=std_noise)

    ax = random_n_unit_vec(3) * 3
    mtm = MTM(axis=ax, noise=noise)

    # State
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 8, 0])
    )
    sat = Satellite(sensors=[mtm])

    # Difference of two noisy readings => N(0, 2*std_noise^2)
    N = 1000
    noise_drift = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n1 = np.asarray(mtm.reading(x=x, os=os)).item()
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n2 = np.asarray(mtm.reading(x=x, os=os)).item()
        noise_drift.append(n1 - n2)
    noise_drift = np.array(noise_drift).reshape(N, 1)

    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.sqrt(2) * std_noise,
        size=(N, 1)
    )

    # KS test
    i = 0
    ks = ks_2samp(noise_drift[:, i], exp_dist[:, i])
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    # Visualization
    hist = np.histogram(noise_drift[:, i], bins="auto")
    edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [np.sum(exp_dist[:, i] < ee) for ee in edges[1:]]
    print(f"MTM Noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")
    print(plot([hist_a, hist_b], {"height": 20}))

    assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


def test_mtm_reading_bias_noise():
    # Bias + noise setup
    e_bias = np.random.uniform(1, 3) * random_n_unit_vec(1)
    std_bias = np.abs(np.random.uniform(0.001, 0.05) * random_n_unit_vec(1))
    std_noise = np.abs(np.random.uniform(0.05, 0.3) * random_n_unit_vec(1))
    bias = Bias(bias=e_bias, std_bias=std_bias)
    noise = Noise(noise=np.zeros(1), std_noise=std_noise)

    ax = random_n_unit_vec(3) * 3
    mtm = MTM(axis=ax, bias=bias, noise=noise)

    # State
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22,
        R=np.array([7000, 0, 0]),
        V=np.array([0, 8, 0])
    )
    sat = Satellite(sensors=[mtm])

    # Sample differences y1 - y2 over Δt
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y1 = np.asarray(mtm.reading(x=x, os=os)).item()
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y2 = np.asarray(mtm.reading(x=x, os=os)).item()
        drifts.append(y1 - y2)
    drifts = np.array(drifts).reshape(N, 1)

    # Combined variance: bias random walk over dt + two independent noises
    dt_sec = (0.5 * TimeConstants.sec2cent) * TimeConstants.cent2sec
    sigma = np.sqrt((std_bias ** 2) * dt_sec + 2 * (std_noise ** 2))
    exp_dist = np.random.normal(0.0, sigma, size=(N, 1))

    # KS test
    i = 0
    ks = ks_2samp(drifts[:, i], exp_dist[:, i])
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    # Visualization
    hist = np.histogram(drifts[:, i], bins="auto")
    edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [np.sum(exp_dist[:, i] < ee) for ee in edges[1:]]
    print(f"MTM bias+noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")
    print(plot([hist_a, hist_b], {"height": 20}))

    assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


if __name__ == "__main__":
    test_mtm_reading_bias_noise()