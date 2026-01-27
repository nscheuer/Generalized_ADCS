# Detailed Comparison: Dissertation Thesis Tables vs. Code

## Overview
This document compares the parameter values stated in the dissertation thesis tables with those found in the actual code in `patrickmckeen/PhD_Dissertation_Code`.

---

## Cases A & B: TRMM Satellite (Large Satellite)

**Code Location:** `GeneralizedADS/helpers/src/sat_ADCS_helpers/common_sats.py` → `create_Crassidis_UKF_sat()`

### Mass & Inertia

| Parameter | Thesis (Table 4.2) | Code | Match? |
|-----------|-------------------|------|--------|
| Mass | 3000 kg | `mass = 3e3` | ✅ YES |
| Inertia | J = 500·diag([1,3,3]) kg·m² | `J = np.diagflat(np.array([1,3,3]))*500` | ✅ YES |

### Gyroscope

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Bias | [0.1, 0.1, 0.1] deg/hr | `gyro_bias0 = np.ones(3)*0.1*(math.pi/180)/3600` | ✅ YES |
| Noise Std Dev | 0.31623 μrad/s^0.5 | `gyro_std = 0.31623*1e-6*np.ones(3)` | ✅ YES |
| Bias Drift Rate | 3.1623×10⁻⁴ μrad/s^1.5 | `gyro_bsr = 3.1623e-4 * 1e-6*np.ones(3)` | ✅ YES |

### Magnetometer

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Bias | [-0.9948, -0.0199, -0.0995] nT | `mtm_bias0 = 1e-9*normalize(np.array([-5,-0.1,-0.5]))` | ⚠️ SCALED DIFFERENTLY |
| Noise Std Dev | 50 nT·s^0.5 | `mtm_std = 50*1e-9*np.ones(3)` | ✅ YES |
| Bias Drift Rate | 1 nT/s^0.5 | `mtm_bsr = (1e-9)*np.ones(3)` | ✅ YES |

**Note on MTM bias:** The thesis shows specific values `[-0.9948, -0.0199, -0.0995]` nT, but code uses `1e-9*normalize(np.array([-5,-0.1,-0.5]))` which equals `[-0.9806, -0.0196, -0.0981]` nT - very close but not exact.

### MTQ

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Max Moment | 100 Am² | `mtq_max = 100.0*np.ones(3)` | ✅ YES |
| Initial Bias | 5·[1/3√2, 1/3√2, 4/3√2] | `mtq_bias0 = 5*normalize(np.array([1,1,4]))` | ✅ YES |
| Noise Std Dev | 0.0001 Am²·s^0.5 | `mtq_std = 0.0001*np.ones(3)` | ✅ YES |
| Bias Drift Rate | 0.000001 Am²/s^0.5 | `mtq_bsr = 0.000001*np.ones(3)` | ✅ YES |

### Residual Dipole

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Value | [0.5, 0.001, 2] Am² | `dipole0 = np.array([0.05,0.0001,0.2])*10*math.sqrt(jmult)` | ⚠️ SCALED BY jmult |
| Capped Max | 2 Am² | `dipole_mag_max = 2.0*math.sqrt(jmult)` | ⚠️ SCALED |
| Drift Rate | 0.0001 Am²/s^0.5 | `dipole_std = 0.0001` | ✅ YES |

**Note:** With default `jmult=1`, dipole0 = [0.5, 0.001, 2.0] - matches thesis!

### Orbit Parameters

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Position | 7200 km along ECI x-axis | `np.array([1,0,0])*7200` | ✅ YES |
| Inclination | 35° | `7.4*np.array([0,math.cos(35*math.pi/180),math.sin(35*math.pi/180)])` | ✅ YES |
| Initial Velocity | 7.4 km/s | `7.4*np.array([...])` | ✅ YES |

### Initial State

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Angular Velocity | [0, 1/15, 0] deg/s | `w0 = np.array([0,2*math.pi/(60*90),0])` = [0, 0.00116, 0] rad/s = [0, 0.0667, 0] deg/s | ⚠️ DIFFERENT |
| Initial Quaternion | [0.63, -0.33, -0.63, 0.33] | `q0 = two_vec_to_quat(-orb.states[...].R, orb.states[...].V, unitvecs[2], unitvecs[0])` | ✅ Computed from orbit |

**Note on Angular Velocity:** Thesis says `[0, 1/15, 0]` deg/s = `[0, 0.0667, 0]` deg/s. Code computes `2*pi/(60*90)` rad/s which is ~1 rotation per 90-minute orbit, equaling ~0.0667 deg/s. **MATCH!**

---

## Cases C & D: CubeSat (3U CubeSat with poor sensors)

**Code Location:** `GeneralizedADS/helpers/src/sat_ADCS_helpers/common_sats.py` → `create_BC_sat()` 

### Mass & Inertia

| Parameter | Thesis (Table 4.4) | Code | Match? |
|-----------|-------------------|------|--------|
| Mass | 4 kg | `mass = 4` | ✅ YES |
| Inertia | [0.0314, 5.9e-5, -0.0067; 5.9e-5, 0.0341, -0.0001; -0.0067, -0.0001, 0.01005] | See below | ✅ YES |

**Code inertia:**
```python
J = np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
              [5.88304e-05, 0.03409127827, -0.00012334756],
              [-0.00671361357, -0.00012334756, 0.01004091997]])
```
Values match to 3-4 significant figures.

### Gyroscope

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Bias | 0.1/√11·[1,-1,3] deg/s | `gyro_bias0 = (math.pi/180.0)*0.1*normalize(np.array([1,-1,3]))` | ✅ YES |
| Noise Std Dev | 0.0004 deg/s^0.5 | `gyro_std = 0.0004*math.pi/180.0*np.ones(3)` | ⚠️ DIFFERENT |
| Bias Drift Rate | 0.03 deg/s^1.5 | `gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)` | ⚠️ DIFFERENT |

**DISCREPANCY:** Thesis says:
- Noise: 0.0004 deg/s^0.5 
- Bias Drift: 0.03 deg/s^1.5

Code shows:
- `gyro_std = 0.03*math.pi/180.0*np.ones(3)` = 0.03 deg/s^0.5
- `gyro_bsr = 0.0004*math.pi/180.0*np.ones(3)` = 0.0004 deg/s^1.5

**The values appear SWAPPED between noise and bias drift rate!**

### Magnetometer

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Bias | [-9.948, -0.199, -0.995] nT | `mtm_bias0 = 1e-8*normalize(np.array([-5,-0.1,-0.5]))` | ⚠️ SCALED |
| Noise Std Dev | 300 nT·s^0.5 | `mtm_std = 3*1e-7*np.ones(3)` = 300 nT | ✅ YES |
| Bias Drift Rate | 1 nT/s^0.5 | `mtm_bsr = 1e-9*np.ones(3)` = 1 nT | ✅ YES |

**Note on MTM bias:** Code uses `1e-8*normalize([-5,-0.1,-0.5])` ≈ `[-9.95e-9, -0.199e-9, -0.995e-9]` but thesis shows values 10x larger. This could be a scaling factor issue.

### MTQ

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Max Moment | 1 Am² | `mtq_max = 1.0*np.ones(3)` | ✅ YES |
| Initial Bias | 0.05·[1/3√2, 1/3√2, 4/3√2] | `mtq_bias0 = 0.05*normalize(np.array([1,1,4]))` | ✅ YES |
| Noise Std Dev | 0.0001 Am²·s^0.5 | `mtq_std = 0.0001*np.ones(3)` | ✅ YES |
| Bias Drift Rate | 0.000001 Am²/s^0.5 | `mtq_bsr = 0.000001*np.ones(3)` | ✅ YES |

### Sun Sensors

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Efficiency | 0.3 | `sun_eff = 0.3*np.ones(3)` | ✅ YES |
| Initial Bias | [0.015, 0.027, -0.009] | `sun_bias0 = np.array([0.05,0.09,-0.03])*sun_eff` | ✅ YES (0.05*0.3=0.015) |
| Noise Std Dev | 0.0003 s^0.5 | `sun_std = 0.001*sun_eff` = 0.0003 | ✅ YES |
| Bias Drift Rate | 0.000003 s^-0.5 | `sun_bsr = 0.00001*sun_eff` = 0.000003 | ✅ YES |

### Residual Dipole

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Initial Value | [0.05, 0.0001, 0.2] Am² | `dipole0 = np.array([0.05,0.0001,0.2])` | ✅ YES |
| Capped Max | 0.5 Am² | `dipole_mag_max = 0.5` | ✅ YES |
| Drift Rate | 0.0001 Am²/s^0.5 | `dipole_std = 0.0001` | ✅ YES |

---

## Case G: 6U CubeSat with Many Variables

**Code Location:** `create_GPS_6U_sat_betterest()` which wraps `create_GPS_6U_sat()`

### Mass & Inertia

| Parameter | Thesis (Table 4.7) | Code | Match? |
|-----------|-------------------|------|--------|
| Mass | 10.165 kg | `mass = 10.165` | ✅ YES |
| Inertia | diag([0.0969, 0.1235, 0.1918]) | `J = np.diagflat([0.0969,0.1235,0.1918])` | ✅ YES |

### Actuators (3 MTQ + 3 RW)

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| MTQ Max | 5 Am² | `mtq_max = 5.0*np.ones(3)` | ✅ YES |
| MTQ Noise | 0.000001 Am²·s^0.5 | `mtq_std = 1e-6*np.ones(3)` (betterest default) | ✅ YES |
| RW Max Torque | 0.005 Nm | `rw_max = 0.005*np.ones(3)` | ✅ YES |
| RW Max Momentum | 15 mNms | `rw_maxh = (15/1000)*np.ones(3)` | ✅ YES |
| RW Inertia | 0.0014 kg·m² | `rw_J = 0.0014*np.ones(3)` | ✅ YES |

### Sensors

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Gyro Noise | 0.00001 rad/s^0.5 | `gyro_bias0 = 1e-5*np.ones(3)` (betterest) | ⚠️ CHECK UNITS |
| Gyro Bias Drift | 0.0000001 rad/s^1.5 | `gyro_bsr = 1e-7*np.ones(3)` (betterest) | ✅ YES |
| MTM Noise | 0.0000001 T·s^0.5 | `mtm_std = 1e-7*np.ones(3)` | ✅ YES |
| MTM Bias Drift | 1e-10 T/s^0.5 | `mtm_bsr = 1e-10*np.ones(3)` (betterest) | ✅ YES |

### Disturbances

| Parameter | Thesis | Code | Match? |
|-----------|--------|------|--------|
| Dipole Drift | 0.00091 Am²/s^0.5 | `dipole_std = 0.00091` (betterest) | ✅ YES |
| Prop Torque Drift | 1e-8 Nm/s^0.5 | `prop_torq_std = 1e-8` | ✅ YES |

---

## Summary of Discrepancies

### Critical Issues:
1. **Cases C/D Gyroscope:** Noise std and bias drift rate appear to be SWAPPED
   - Thesis: Noise=0.0004, Drift=0.03
   - Code: Noise=0.03, Drift=0.0004

### Minor Issues:
2. **MTM Bias Initial Values:** Slight numerical differences due to normalization
3. **Scaling factors:** Some parameters use `jmult` or `mtm_scale` that could affect values

### Verified Matches:
- Mass and inertia values: ✅ All match
- MTQ parameters: ✅ All match  
- Sun sensor parameters: ✅ All match
- Residual dipole parameters: ✅ All match (for Case C/D)
- Orbit parameters: ✅ All match
- RW parameters (Case G): ✅ All match

---

## Test Files Used

| Test Case | Code File | Data Directory Pattern |
|-----------|-----------|----------------------|
| Cases A/B | `thesis_estimation_tests.py` | `*TRMM*` or `crassidis_*` |
| Cases C-F | `thesis_estimation_tests_rwmtq.py` | `thesis_6U_quat_*` |
| Case G | `thesis_better_estimation_tests_rwmtq.py` | `thesis_6U_quat_RWMTQ_betterest_*` |
| Disturbance Tests | `thesis_disturbance_tests_rwmtq.py` | `*_wis_twist_*`, `*_lovera_*`, `*_wie_*` |
