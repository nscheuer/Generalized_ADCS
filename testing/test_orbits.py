import sys
import os
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

def test_orbit_creation():
    n_n = random_n_unit_vec(3)
    ecc = np.random.uniform(0.1, 0.4)
    rp = np.random.uniform(6800, 9000)
    a = rp/(1 - ecc)
    vp = np.sqrt(EarthConstants.mu_e*(1+ecc)/(1-ecc)/a)
    h = rp*vp
    th = np.random.uniform(0, 2*np.pi)
    r = h*h/(EarthConstants.mu_e*(1+ecc*np.cos(th)))
    v = np.sqrt(EarthConstants.mu_e*(2/r - 1/a))

    n_rp = normalize(np.cross(random_n_unit_vec(3),n_n))
    n_vp = normalize(np.cross(n_n,n_rp))

    rvec = r*(n_rp*np.cos(th)+n_vp*np.sin(th))
    cphi = (1+ecc*np.cos(th))/np.sqrt(1+ecc*ecc+2*ecc*np.cos(th))
    phi = np.arccos(cphi)*np.sign(np.sin(th))
    psi = th + 0.5*np.pi - phi
    vvec = v*(n_rp*np.cos(psi) + n_vp*np.sin(psi))
    zero_time = 0.22

    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=zero_time, R=rvec, V=vvec)
    os0 = os.copy()
    orb0 = Orbit(os0=os)

    dt = 3600
    N_dt = 24*5
    
    end_time = zero_time + TimeConstants.sec2cent*dt*N_dt
    print("Creating Orbit 1")
    orb1 = Orbit(os0=os, end_time=end_time, dt=dt, fast=False)
    print("Creating Orbit 2")
    orb2 = Orbit([*orb1.states.values()])
    print("Creating Orbit 3")
    orb3 = Orbit(os0=os, end_time=end_time, dt=dt, fast=True)
    Bvecs = orb3.get_b_eci_orbit()
    for j in range(len(orb3.times)):
        orb3.states[orb3.times[j]].B = Bvecs[j,:]
   
    # Check times
    print("Checking Orbit Times")
    assert orb0.times == zero_time
    assert np.allclose(orb1.times , [zero_time+TimeConstants.sec2cent*dt*j for j in range(N_dt + 1)])
    assert np.allclose(orb2.times , [zero_time+TimeConstants.sec2cent*dt*j for j in range(N_dt+ 1)])
    assert np.allclose(orb3.times , [zero_time+TimeConstants.sec2cent*dt*j for j in range(N_dt+ 1)])

    # Check orbit states
    print("Checking Orbit States")
    assert np.allclose(os0.R,orb0.states[zero_time].R)
    assert np.allclose(os0.V,orb0.states[zero_time].V)
    assert np.allclose(os0.J2000,orb0.states[zero_time].J2000)
    assert np.allclose(zero_time,orb0.states[zero_time].J2000)
    orb0statelist = [*orb0.states.values()]
    assert len(orb0statelist)==1
    assert np.allclose(os0.R,[j.R for j in orb0statelist][0])
    assert np.allclose(os0.V,[j.V for j in orb0statelist][0])
    assert np.allclose(os0.J2000,[j.J2000 for j in orb0statelist][0])

    ind = 0
    print("Checking Orbit 1 Manual Propagation")
    for t in orb1.times:
        assert orb1.states[t].J2000 == t
        assert t == zero_time + TimeConstants.sec2cent*dt*ind

        if ind>0:
            os = orb1.states[t]
            test_os = prev_os.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True, fast=True)
            assert np.isclose( test_os.J2000 , os.J2000, rtol=1e-15, atol=1e-15)
            assert np.allclose(test_os.R , os.R)
            assert np.allclose(test_os.V , os.V)
        prev_os = orb2.states[t]
        ind += 1

    ind = 0
    print("Checking Orbit 3 Manual Propagation")
    for t in orb3.times:
        assert orb3.states[t].J2000 == t
        assert t == zero_time + TimeConstants.sec2cent*dt*ind

        if ind>0:
            os = orb3.states[t]
            test_os = prev_os.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True, fast=True)
            assert np.isclose( test_os.J2000 , os.J2000, rtol=1e-15, atol=1e-15)
            assert np.allclose(test_os.R , os.R)
            assert np.allclose(test_os.V , os.V)
        prev_os = orb3.states[t]
        ind += 1

    assert np.allclose(os0.B,orb0statelist[0].B)
    assert np.allclose(os0.S,orb0statelist[0].S)
    assert np.allclose(os0.rho,orb0statelist[0].rho)

    print("Checking Orbit 1 Constructor")
    for t in orb1.times:
        assert np.allclose(orb1.states[t].B,orb1.states[t].get_b_eci())
        assert np.allclose(orb1.states[t].S,orb1.states[t].get_sun_eci())
        osbackup = Orbital_State(ephem=ephem, J2000=t, R=orb1.states[t].R, V=orb1.states[t].V)
        assert np.allclose(orb1.states[t].rho,osbackup.rho)
        assert np.allclose(orb1.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb1.states[t].S,osbackup.get_sun_eci())
        assert np.allclose(orb1.states[t].TAI,osbackup.TAI)
        assert np.allclose(orb1.states[t].LLA,osbackup.LLA)
        assert np.allclose(orb1.states[t].ECEF,osbackup.ECEF)
        assert orb1.states[t].datetime == osbackup.datetime
        assert np.allclose(orb1.states[t].geocentric,osbackup.geocentric)
        assert np.allclose(orb1.states[t].ECI2ENUmat,osbackup.ECI2ENUmat)

    print("Checking Orbit 1 Vector Length")
    vecs1 = orb1.get_vecs()
    assert len(vecs1[0]) == 121
    assert len(vecs1[1]) == 121
    assert len(vecs1[2]) == 121
    assert len(vecs1[3]) == 121
    assert len(vecs1[4]) == 121
    assert np.all([np.allclose(vecs1[0][j],orb1.states[orb1.times[j]].R) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[1][j],orb1.states[orb1.times[j]].V) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[2][j],orb1.states[orb1.times[j]].B) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[3][j],orb1.states[orb1.times[j]].S) for j in range(len(vecs1[0]))])
    assert np.all([np.allclose(vecs1[4][j],orb1.states[orb1.times[j]].rho) for j in range(len(vecs1[0]))])

    print("Checking Orbit 2 Constructor")
    for t in orb2.times:
        B_saved = orb2.states[t].B
        B_computed = orb2.states[t].get_b_eci()
        assert np.allclose(orb2.states[t].B,orb2.states[t].get_b_eci())
        assert np.allclose(orb2.states[t].S,orb2.states[t].get_sun_eci())
        osbackup = Orbital_State(ephem=ephem, J2000=t, R=orb2.states[t].R, V=orb2.states[t].V)
        assert np.allclose(orb2.states[t].rho,osbackup.rho)
        assert np.allclose(orb2.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb2.states[t].S,osbackup.get_sun_eci())
        assert np.allclose(orb2.states[t].TAI,osbackup.TAI)
        assert np.allclose(orb2.states[t].LLA,osbackup.LLA)
        assert np.allclose(orb2.states[t].ECEF,osbackup.ECEF)
        assert orb2.states[t].datetime == osbackup.datetime
        assert np.allclose(orb2.states[t].geocentric,osbackup.geocentric)
        assert np.allclose(orb2.states[t].ECI2ENUmat,osbackup.ECI2ENUmat)

    print("Checking Orbit 2 Vector Length")
    vecs2 = orb2.get_vecs()
    assert len(vecs2[0]) == 121
    assert len(vecs2[1]) == 121
    assert len(vecs2[2]) == 121
    assert len(vecs2[3]) == 121
    assert len(vecs2[4]) == 121
    assert np.all([np.allclose(vecs2[0][j],orb2.states[orb2.times[j]].R) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[1][j],orb2.states[orb2.times[j]].V) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[2][j],orb2.states[orb2.times[j]].B) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[3][j],orb2.states[orb2.times[j]].S) for j in range(len(vecs2[0]))])
    assert np.all([np.allclose(vecs2[4][j],orb2.states[orb2.times[j]].rho) for j in range(len(vecs2[0]))])

    print("Checking Orbit 3 Constructor")
    for t in orb3.times:
        assert np.allclose(orb3.states[t].B,orb3.states[t].get_b_eci())
        osbackup = Orbital_State(ephem=ephem, J2000=t, R=orb3.states[t].R, V=orb3.states[t].V)
        assert np.allclose(orb3.states[t].rho,osbackup.rho)
        assert np.allclose(orb3.states[t].B,osbackup.get_b_eci())
        assert np.allclose(orb3.states[t].TAI,osbackup.TAI)
        assert np.allclose(orb3.states[t].ECEF,osbackup.ECEF)
        assert orb3.states[t].datetime == osbackup.datetime
        assert np.allclose(orb3.states[t].geocentric,osbackup.geocentric)

    print("Checking Orbit 3 Vector Length")
    vecs3 = orb3.get_vecs()
    assert len(vecs3[0]) == 121
    assert len(vecs3[1]) == 121
    assert len(vecs3[2]) == 121
    assert len(vecs3[3]) == 121
    assert len(vecs3[4]) == 121
    assert np.all([np.allclose(vecs3[0][j],orb3.states[orb3.times[j]].R) for j in range(len(vecs3[0]))])
    assert np.all([np.allclose(vecs3[1][j],orb3.states[orb3.times[j]].V) for j in range(len(vecs3[0]))])
    assert np.all([np.allclose(vecs3[2][j],orb3.states[orb3.times[j]].B) for j in range(len(vecs3[0]))])
    assert np.all([np.allclose(vecs3[3][j],orb3.states[orb3.times[j]].S) for j in range(len(vecs3[0]))])
    assert np.all([np.allclose(vecs3[4][j],orb3.states[orb3.times[j]].rho) for j in range(len(vecs3[0]))])

    print("Checking Orbit 0 next_state()")
    test0 = orb0.get_os(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test0 = orb0.next_state(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test0 = orb0.next_state(os0)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    print("Checking Orbit 1 next_state()")
    test1 = orb1.get_os(zero_time)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    test1 = orb1.next_state(zero_time)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    test1 = orb1.next_state(os0)
    assert test1.J2000 == zero_time
    assert np.allclose(os0.R,test1.R)
    assert np.allclose(os0.V,test1.V)
    assert np.allclose(os0.B,test1.B)
    assert np.allclose(os0.S,test1.S)
    assert np.allclose(os0.rho,test1.rho)

    print("Checking Orbit 2 next_state()")
    test2 = orb2.get_os(zero_time)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)

    test2 = orb2.next_state(zero_time)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)

    test2 = orb2.next_state(os0)
    assert test2.J2000 == zero_time
    assert np.allclose(os0.R,test2.R)
    assert np.allclose(os0.V,test2.V)
    assert np.allclose(os0.B,test2.B)
    assert np.allclose(os0.S,test2.S)
    assert np.allclose(os0.rho,test2.rho)

    print("Checking Orbit 3 next_state()")
    test3 = orb3.get_os(zero_time)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)

    test3 = orb3.next_state(zero_time)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)

    test3 = orb3.next_state(os0)
    assert test3.J2000 == zero_time
    assert np.allclose(os0.R,test3.R)
    assert np.allclose(os0.V,test3.V)
    assert np.allclose(os0.B,test3.B)
    assert np.allclose(os0.S,test3.S)
    assert np.allclose(os0.rho,test3.rho)


    #last match
    test0 = orb0.get_os(zero_time)
    assert test0.J2000 == zero_time
    assert np.allclose(os0.R,test0.R)
    assert np.allclose(os0.V,test0.V)
    assert np.allclose(os0.B,test0.B)
    assert np.allclose(os0.S,test0.S)
    assert np.allclose(os0.rho,test0.rho)

    test1 = orb1.get_os(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(orb1.states[orb1.times[-1]])
    test1a = orb1.states[orb1.times[-1]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test2 = orb2.get_os(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(orb2.states[orb2.times[-1]])
    test2a = orb2.states[orb2.times[-1]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)


    test3 = orb3.get_os(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(zero_time+TimeConstants.sec2cent*dt*N_dt)
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(orb3.states[orb3.times[-1]])
    test3a = orb3.states[orb3.times[-1]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)


    #middle one
    test1 = orb1.get_os(zero_time+TimeConstants.sec2cent*dt*24*2)
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(zero_time+TimeConstants.sec2cent*dt*24*2)
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test1 = orb1.next_state(orb1.states[orb1.times[24*2]])
    test1a = orb1.states[orb1.times[24*2]]
    assert test1a.J2000 == test1.J2000
    assert np.allclose(test1a.R,test1.R)
    assert np.allclose(test1a.V,test1.V)
    assert np.allclose(test1a.B,test1.B)
    assert np.allclose(test1a.S,test1.S)
    assert np.allclose(test1a.rho,test1.rho)

    test2 = orb2.get_os(zero_time+TimeConstants.sec2cent*dt*24*2)
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(zero_time+TimeConstants.sec2cent*dt*24*2)
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test2 = orb2.next_state(orb2.states[orb2.times[24*2]])
    test2a = orb2.states[orb2.times[24*2]]
    assert test2a.J2000 == test2.J2000
    assert np.allclose(test2a.R,test2.R)
    assert np.allclose(test2a.V,test2.V)
    assert np.allclose(test2a.B,test2.B)
    assert np.allclose(test2a.S,test2.S)
    assert np.allclose(test2a.rho,test2.rho)

    test3 = orb3.get_os(zero_time+TimeConstants.sec2cent*dt*24*2)
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(zero_time+TimeConstants.sec2cent*dt*24*2)
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

    test3 = orb3.next_state(orb3.states[orb3.times[24*2]])
    test3a = orb3.states[orb3.times[24*2]]
    assert test3a.J2000 == test3.J2000
    assert np.allclose(test3a.R,test3.R)
    assert np.allclose(test3a.V,test3.V)
    assert np.allclose(test3a.B,test3.B)
    assert np.allclose(test3a.S,test3.S)
    assert np.allclose(test3a.rho,test3.rho)

if __name__ == "__main__":
    test_orbit_creation()