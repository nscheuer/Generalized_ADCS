import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from typing import List
from scipy.stats import kstest, ks_2samp
from asciichartpy import plot

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.sensors import SunSensor
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, normalize
from ADCS.helpers.math_constants import MathConstants

def test_sun_reading_etc_clean():
    ax = random_n_unit_vec(3)*3
    efficiency = 0.3
    sun = SunSensor(axis=ax, efficiency=efficiency)
    assert sun.sample_time == 0.1
    assert np.all(sun.bias.bias == np.zeros(1))
    assert np.all(sun.bias.std_bias == np.zeros(1))
    assert np.all(sun.noise.noise == np.zeros(1))
    assert np.all(sun.noise.std_noise == np.zeros(1))

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    x = np.concatenate([w0, q0])
    sat = Satellite(sensors=[sun])
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))
    vecs = os.get_state_vector(x=x)

    out = sun.reading(x=x, os=os)
    nSB = normalize(vecs["s"] - vecs["r"])
    expected = max(0, np.dot(ax/3, nSB)*efficiency)
    assert np.allclose(out, expected)

    xfun = lambda c: sun.reading(x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
    bfun = lambda c: SunSensor(axis=ax, efficiency=efficiency).reading(x=x, os=os)

    Jxfun = nd.Jacobian(xfun)(x.flatten().tolist())
    Jbfun = nd.Jacobian(bfun)(20000)

    assert np.allclose(bfun(20000) , expected)
    assert np.allclose(xfun(x) , expected)
    assert np.allclose(Jxfun.T, sun.basestate_jac(x=x, os=os))
    assert np.allclose(Jbfun, sun.bias_jac(x=x, os=os))
    assert np.all(np.isclose( sun.bias_jac(x=x, os=os) , 1))


def test_sun_reading_bias_KS():
    # --- Bias-only setup ---------------------------------------------------
    e_bias = np.random.uniform(1, 3) * random_n_unit_vec(1)
    std_bias = np.abs(np.random.uniform(0.001, 0.1) * random_n_unit_vec(1))
    bias = Bias(bias=e_bias, std_bias=std_bias)

    ax = random_n_unit_vec(3) * 3
    efficiency = 0.3
    sun = SunSensor(axis=ax, efficiency=efficiency, bias=bias)

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    sat = Satellite(sensors=[sun])

    # Repeatedly advance time and record bias drift
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b1 = sun.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b2 = sun.reading(x=x, os=os)
        drifts.append(b1 - b2)

    drifts = np.array(drifts).reshape(N, 1)  # shape (N,1)

    # Expected normal samples (per-component)
    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.abs(std_bias) * np.sqrt(0.5),
        size=(N, 1)
    )

    # KS test (single component)
    i = 0
    ks = kstest(drifts[:, i], exp_dist[:, i])
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    # Optional visualization
    hist = np.histogram(drifts[:, i], bins="auto")
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
    graph_data = [hist_a, hist_b]
    print(f"SunSensor Bias component {i} KS:")
    print(plot(graph_data, {"height": 20}))

    assert ks.pvalue > 0.1 or np.abs(ks.statistic) < threshold


def test_sun_reading_noise_KS():
    # --- Noise-only setup --------------------------------------------------
    std_noise = np.abs(np.random.uniform(0.001, 0.1, size=1))
    noise = Noise(noise=np.zeros(1), std_noise=std_noise)

    ax = random_n_unit_vec(3) * 3
    efficiency = 0.3
    sun = SunSensor(axis=ax, efficiency=efficiency, noise=noise)

    assert sun.sample_time == 0.1
    assert np.allclose(sun.noise.noise, np.zeros(1))
    assert np.allclose(sun.noise.std_noise, std_noise)
    assert np.allclose(sun.bias.bias, np.zeros(1))
    assert np.allclose(sun.bias.std_bias, np.zeros(1))

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    sat = Satellite(sensors=[sun])

    # Each reading call adds new Gaussian noise; Δn = n1 - n2 ~ N(0, 2σ²)
    N = 1000
    noise_drift = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n1 = sun.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n2 = sun.reading(x=x, os=os)
        noise_drift.append(n1 - n2)

    noise_drift = np.stack(noise_drift, axis=0).reshape(N, 1)  # shape (N,1)

    # Expected distribution for Δn
    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.sqrt(2) * std_noise,
        size=(N, 1)
    )

    # KS Test
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    i = 0
    ks = ks_2samp(noise_drift[:, i], exp_dist[:, i])
    print(f"SunSensor Noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")

    # Optional visualization
    hist = np.histogram(noise_drift[:, i], bins="auto")
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
    graph_data = [hist_a, hist_b]
    print(plot(graph_data, {"height": 20}))

    assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


def test_sun_reading_bias_noise():
    # --- Bias and noise setup ---------------------------------------------
    e_bias = np.random.uniform(1, 3) * random_n_unit_vec(1)
    std_bias = np.abs(np.random.uniform(0.001, 0.05) * random_n_unit_vec(1))
    std_noise = np.abs(np.random.uniform(0.05, 0.3) * random_n_unit_vec(1))

    bias = Bias(bias=e_bias, std_bias=std_bias)
    noise = Noise(noise=np.zeros(1), std_noise=std_noise)

    ax = random_n_unit_vec(3) * 3
    efficiency = 0.3
    sun = SunSensor(axis=ax, efficiency=efficiency, bias=bias, noise=noise)

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    sat = Satellite(sensors=[sun])

    # --- Sample differences over Δt ---------------------------------------
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y1 = sun.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y2 = sun.reading(x=x, os=os)
        drifts.append(y1 - y2)

    drifts = np.stack(drifts, axis=0).reshape(N, 1)  # shape (N,1)

    # --- Expected combined distribution -----------------------------------
    # Δt in seconds (0.5 s between half-steps)
    dt_sec = (0.5 * TimeConstants.sec2cent) * TimeConstants.cent2sec

    # Effective standard deviation per component:
    sigma = np.sqrt((std_bias**2) * dt_sec + 2 * (std_noise**2))

    exp_dist = np.random.normal(0.0, sigma, size=(N, 1))

    # --- KS Test -----------------------------------------------------------
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    i = 0
    ks = ks_2samp(drifts[:, i], exp_dist[:, i])
    print(f"SunSensor bias+noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")

    # Optional visualization
    hist = np.histogram(drifts[:, i], bins="auto")
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
    graph_data = [hist_a, hist_b]
    print(plot(graph_data, {"height": 20}))

    assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold

if __name__ == "__main__":
    test_sun_reading_etc_clean()
    test_sun_reading_bias_KS()
    test_sun_reading_noise_KS()
    test_sun_reading_bias_noise()