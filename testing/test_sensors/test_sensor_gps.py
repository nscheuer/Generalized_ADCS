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
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
from ADCS.helpers.math_constants import MathConstants

def test_gps_reading_etc_clean():
    gps = GPS()
    assert gps.sample_time == 0.1
    assert np.all(gps.bias.bias == np.zeros(6))
    assert np.all(gps.bias.std_bias == np.zeros(6))
    assert np.all(gps.noise.noise == np.zeros(6))
    assert np.all(gps.noise.std_noise == np.zeros(6))

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    x = np.concatenate([w0, q0])
    sat = Satellite(sensors=[gps])
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    out = gps.reading(x=x, os=os)
    assert np.allclose(out, np.concatenate([os.ECEF, os.eci_to_ecef(os.V)]))


    Rr = random_n_unit_vec(3)*np.random.uniform(6800, 9000)
    Vv = random_n_unit_vec(3)*np.random.uniform(4, 20)
    os = Orbital_State(ephem=ephem, J2000=0.22, R=Rr, V=Vv)

    out = gps.reading(x=x, os=os)
    assert np.allclose(out, np.concatenate([os.ECEF, os.eci_to_ecef(os.V)]))


    xfun = lambda c: gps.reading(x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
    bfun = lambda c: GPS().reading(x=x, os=os)

    Jxfun = np.array(nd.Jacobian(xfun)(x.flatten().tolist())).T
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T

    assert np.allclose(bfun(20000) , np.concatenate([os.ECEF,os.eci_to_ecef(os.V)]))
    assert np.allclose(xfun(x) , np.concatenate([os.ECEF,os.eci_to_ecef(os.V)]))
    assert np.allclose(Jxfun, gps.basestate_jac(x=x, os=os))
    assert np.allclose(Jbfun, gps.bias_jac(x=x, os=os))
    assert np.all(np.isclose( gps.bias_jac(x=x, os=os) , np.zeros((0,6))))
    assert np.all(np.isclose( gps.basestate_jac(x=x, os=os) , np.zeros((7,6)) ))


def test_gps_reading_etc_bias():
    e_bias = np.random.uniform(1, 3)*random_n_unit_vec(6)
    std_bias = np.abs(np.random.uniform(0.001, 0.1)*random_n_unit_vec(6))
    bias = Bias(bias=e_bias, std_bias=std_bias)

    gps = GPS(bias=bias)
    assert gps.sample_time == 0.1
    assert np.all(gps.bias.bias == e_bias)
    assert np.all(gps.bias.std_bias == std_bias)
    assert np.all(gps.noise.noise == np.zeros(6))
    assert np.all(gps.noise.std_noise == np.zeros(6))

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    x = np.concatenate([w0, q0])
    sat = Satellite(sensors=[gps])
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    out = gps.reading(x=x, os=os)
    assert np.allclose(out, np.concatenate([os.ECEF, os.eci_to_ecef(os.V)])+e_bias)


    xfun = lambda c: gps.reading(x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
    bfun = lambda c: GPS(bias=Bias(bias=c, std_bias=std_bias)).reading(x=x, os=os)

    Jxfun = np.array(nd.Jacobian(xfun)(x.flatten().tolist()))
    Jbfun = np.array(nd.Jacobian(bfun)(e_bias.flatten().tolist()))

    assert np.allclose(bfun(e_bias), np.concatenate([os.ECEF,os.eci_to_ecef(os.V)])+e_bias)
    assert np.allclose( xfun(x) , np.concatenate([os.ECEF,os.eci_to_ecef(os.V)])+e_bias)

    assert np.allclose(Jxfun.T, gps.basestate_jac(x=x, os=os))
    assert np.allclose(Jbfun, gps.bias_jac(x=x, os=os))

    assert np.all(np.isclose( gps.bias_jac(x=x, os=os) , np.eye(6) ))
    assert np.all(np.isclose( gps.basestate_jac(x=x, os=os) , np.zeros((7,6))))


    N = 1000
    drifts = []
    # Repeatedly advance time and record bias drift
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b1 = gps.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        b2 = gps.reading(x=x, os=os)
        drifts.append(b1 - b2)

    drifts = np.array(drifts)  # shape (N,6)

    # Expected normal samples (per-component)
    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.abs(std_bias) * np.sqrt(0.5),
        size=(N, 6)
    )

    # KS test for each component
    for i in range(6):
        ks = kstest(drifts[:, i], exp_dist[:, i])
        threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

        # Optional visualization (if plot() available)
        hist = np.histogram(drifts[:, i], bins="auto")
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
        graph_data = [hist_a, hist_b]
        print(f"GPS Bias component {i} KS:")
        print(plot(graph_data, {"height": 20}))

        assert ks.pvalue > 0.1 or np.abs(ks.statistic) < threshold


def test_gps_reading_etc_noise():
    # --- Setup ------------------------------------------------------------
    std_noise = np.abs(np.random.uniform(0.001, 0.1, size=6))
    noise = Noise(noise=np.zeros(6), std_noise=std_noise)

    gps = GPS(noise=noise)

    assert gps.sample_time == 0.1
    assert np.allclose(gps.noise.noise, np.zeros(6))
    assert np.allclose(gps.noise.std_noise, std_noise)
    assert np.allclose(gps.bias.bias, np.zeros(6))
    assert np.allclose(gps.bias.std_bias, np.zeros(6))

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0])
    )

    sat = Satellite(sensors=[gps])

    # --- Noise sampling ----------------------------------------------------
    # Each reading call adds new Gaussian noise, so consecutive readings differ by
    # Δn = n₁ - n₂, where each nᵢ ~ N(0, σ²). Therefore, Δn ~ N(0, 2σ²).
    N = 1000
    noise_drift = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n1 = gps.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        n2 = gps.reading(x=x, os=os)
        noise_drift.append(n1 - n2)

    noise_drift = np.stack(noise_drift, axis=0)  # shape (N,6)

    # Expected distribution for Δn
    exp_dist = np.random.normal(
        loc=0.0,
        scale=np.sqrt(2) * std_noise,   # sqrt(2) because difference of two normals
        size=(N, 6)
    )

    # --- KS Test -----------------------------------------------------------
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    for i in range(6):
        ks = ks_2samp(noise_drift[:, i], exp_dist[:, i])
        print(f"GPS Noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")

        # Optional visualization if plot() utility available
        hist = np.histogram(noise_drift[:, i], bins="auto")
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
        graph_data = [hist_a, hist_b]
        print(plot(graph_data, {"height": 20}))

        assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


def test_gps_reading_bias_noise():
    # --- Bias and noise setup ---------------------------------------------
    e_bias = np.random.uniform(1, 3) * random_n_unit_vec(6)
    std_bias = np.abs(np.random.uniform(0.001, 0.05) * random_n_unit_vec(6))
    std_noise = np.abs(np.random.uniform(0.05, 0.3) * random_n_unit_vec(6))

    bias = Bias(bias=e_bias, std_bias=std_bias)
    noise = Noise(noise=np.zeros(6), std_noise=std_noise)
    gps = GPS(bias=bias, noise=noise)

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    x = np.concatenate([w0, q0])

    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0])
    )

    sat = Satellite(sensors=[gps])

    # --- Sample differences over Δt ---------------------------------------
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y1 = gps.reading(x=x, os=os)
        os.J2000 += 0.5 * TimeConstants.sec2cent
        y2 = gps.reading(x=x, os=os)
        drifts.append(y1 - y2)

    drifts = np.stack(drifts, axis=0)  # shape (N,6)

    # --- Expected combined distribution -----------------------------------
    # Δt in seconds (0.5 s between half-steps)
    dt_sec = (0.5 * TimeConstants.sec2cent) * TimeConstants.cent2sec

    # Effective standard deviation per component:
    sigma = np.sqrt((std_bias**2) * dt_sec + 2 * (std_noise**2))

    exp_dist = np.random.normal(0.0, sigma, size=(N, 6))

    # --- KS Tests per component -------------------------------------------
    threshold = np.sqrt((1 / N) * -0.5 * np.log(0.5 * 1e-5))

    for i in range(6):
        ks = ks_2samp(drifts[:, i], exp_dist[:, i])
        print(f"GPS bias+noise component {i} KS: p={ks.pvalue:.4f}, stat={ks.statistic:.4f}")

        # Optional visualization
        hist = np.histogram(drifts[:, i], bins="auto")
        hist_edges = hist[1]
        hist_a = np.cumsum(hist[0]).tolist()
        hist_b = [sum(exp_dist[:, i] < ee) for ee in hist_edges[1:]]
        graph_data = [hist_a, hist_b]
        print(plot(graph_data, {"height": 20}))

        assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


if __name__ == "__main__":
    test_gps_reading_etc_noise()