# Complete Thesis Parameter Reference

This document contains ALL verified parameters from the PhD thesis, cross-referenced with the original dissertation code.

**Source Code:** `patrickmckeen/PhD_Dissertation_Code` (cloned to `dissertation_code_temp/`)
**Thesis Location:** `/mnt/c/Users/LV - Patrick McKeen/Writing/Dissertation/McKeen_PhD_Thesis/MIT-thesis-template/`

---

## Chapter 6: Disturbance Control

### Lovera MTQ-PD Controller (Table 6.x)

**Source:** `dissertation_code_temp/GeneralizedADS/control/src/sat_ADCS_control/lovera_pd.py`

| Parameter | Thesis Value | Code Location | Notes |
|-----------|--------------|---------------|-------|
| **Satellite Inertia** | J = diag([27, 17, 25]) kg·m² | `disturbance.tex` | Large satellite test case |
| **ε (epsilon)** | 0.01 | `gain_info[0]` | Stability scaling factor |
| **kp** | 50 | `gain_info[1]` | Proportional gain |
| **kv** | 50 | `gain_info[2]` | Derivative gain |
| **MTQ Max** | 50 Am² | From Lovera 2005 paper | |
| **Orbit** | 450 km, 87° inclination | `lovera_orb_1` file | Circular |
| **Duration** | 10 hours (36000s) | From thesis figures | Required for convergence |

**Control Law:**
```
τ_des = -J⁻¹(ε²·kp·δ₁:₃ + ε·kv·ω)
m_coils = (B × τ_des) / |B|²
```

### Wisniewski Sliding Mode Controller (Table 6.x)

**Source:** `dissertation_code_temp/GeneralizedADS/control/src/sat_ADCS_control/wisniewski_sliding.py`

| Parameter | Thesis Value | Code Location | Notes |
|-----------|--------------|---------------|-------|
| **Satellite Inertia** | J = diag([27, 17, 25]) kg·m² | Same as Lovera | |
| **Ks (Lambda_s)** | 0.003·I₃ | `gain_info[1]` | Sliding gain |
| **Kq (Lambda_q)** | 0.002·I₃ | `gain_info[0]` | Quaternion gain |
| **MTQ Max** | 50 Am² | | |
| **Orbit** | 450 km, 87° inclination | | Circular |
| **Duration** | 10 hours (36000s) | | |

**Note:** Thesis states Kq = (1/λ_avg)·0.004·I₃ where λ_avg = (λ₁+λ₂+λ₃)/3 for eigenvalues of J.
With J = diag([27,17,25]), λ_avg ≈ 23, so Kq ≈ 0.002·I₃.

**Control Law:**
```
s = ω·J + q₁:₃·Λ_q
τ_des = -0.5(q₀·ω_err + q₁:₃×ω_err)·Λ_q + ω×(J·ω + h_RW) - s·Λ_s
u_des = s·(s·τ_des)/(|s|²)
m_coils = (B × u_des) / |B|²
```

### CubeSat Controller Gains

| Controller | Ks | Kq | kp | kv | ε |
|------------|----|----|----|----|---|
| Lovera CubeSat | - | - | 0.01 | 0.015 | 0.001 |
| Wisniewski CubeSat | 0.0003·I₃ | 0.0001·I₃ | - | - | - |
| Modified Sliding | 0.006·I₃ | 0.0002/λ_avg·I₃ | - | - | - |

---

## Chapter 7: Planning

### Spinning Solution Test (Table 7.1 - tab:plan_dist_test_details)

**Source:** `dissertation_code_temp/GeneralizedADS/ADCS/thesis_plan_tests_rwmtq.py`

| Parameter | Thesis Value | Notes |
|-----------|--------------|-------|
| **Satellite Inertia** | J = [[0.1, 0, 0.00013], [0, 0.05, -0.00021], [0.00013, -0.00021, 0.005]] kg·m² | 3U CubeSat with off-diagonal terms |
| **MTQ Max (x)** | 0.19 Am² | |
| **MTQ Max (y,z)** | 0.57 Am² | |
| **RW Axis** | [0, 1, 0] (y-axis) | Single reaction wheel |
| **RW Max Torque** | 0.2 mNm | |
| **RW Max Momentum** | 2 mNms | |
| **RW Inertia** | 2×10⁻⁶ kg·m² | |
| **Disturbance** | [0.3, 0, 0] mNm | Body-fixed propulsion torque |
| **Orbit Radius** | 6800 km | Near-circular |
| **Goal** | Point z-axis anti-ram | Reduced attitude goal |
| **Initial q** | [-0.232, -0.664, -0.234, -0.671] | From thesis figure |
| **Initial ω** | [0, 0, 0] rad/s | |

### Monte Carlo Test (Table 7.2 - tab:mc_sat_params)

**Source:** `dissertation_code_temp/GeneralizedADS/ADCS/thesis_plan_tests_rwmtq.py`

| Parameter | Thesis Value | Notes |
|-----------|--------------|-------|
| **Satellite Jxx** | 0.005256 kg·m² | |
| **Satellite Jyy = Jzz** | 0.04939 kg·m² | Axisymmetric |
| **MTQ Max (x)** | 0.19 Am² | |
| **RW Max Torque** | 0.0002 Nm | 0.2 mNm |
| **RW Momentum** | 0.002 Nms | 2 mNms |
| **RW Inertia** | 2×10⁻⁶ kg·m² | |
| **RW Axis** | [0, 1, 0] | y-axis |
| **Orbit** | 429 km, 51.5° inclination | ISS orbit |
| **Initial q** | [0, 0, 1, 0] | 180° slew start |
| **Goal q** | [0, 1, 0, 0] | 180° slew target |
| **Initial ω** | [0, 0, 0] rad/s | |
| **Initial h_RW** | 0 Nms | |
| **Trials** | 100 | |

### Sequential Planning Test (Table 7.6 - tab:seq_test_details)

**Source:** `dissertation_code_temp/GeneralizedADS/ADCS/thesis_plan_tests_rwmtq.py`

| Parameter | Thesis Value | Notes |
|-----------|--------------|-------|
| **Satellite** | 6U CubeSat (ASTERIA-like) | |
| **Actuators** | 3 RW only | No MTQ |
| **Goal Sequence** | 5 goals | See below |
| **Trajectory Duration** | 450s each | With 150s overlap |
| **Precalculation Time** | 15s | |

**Goal Timeline:**
| Time (s) | Body Axis | World Direction |
|----------|-----------|-----------------|
| 0-150 | -x | anti-ram |
| 150-1100 | -x | anti-ram |
| 1200-1500 | z | nadir |
| 1600-1900 | z | zenith |
| 2000-2400 | z | orbit_normal |
| 2500+ | -x | anti-ram |

---

## Chapter 4: Estimation

### Cases A & B: TRMM Satellite (Table 4.2)

**Factory Function:** `create_Crassidis_UKF_sat()` in `common_sats.py`

| Parameter | Thesis Value | Code Match |
|-----------|--------------|------------|
| Mass | 3000 kg | ✅ `mass = 3e3` |
| Inertia | 500·diag([1,3,3]) kg·m² | ✅ |
| Gyro Bias Init | [0.1,0.1,0.1] deg/hr | ✅ |
| Gyro Noise σ | 0.31623 μrad/s^0.5 | ✅ |
| Gyro Drift σ | 3.1623e-4 μrad/s^1.5 | ✅ |
| MTM Noise σ | 50 nT·s^0.5 | ✅ |
| MTQ Max | 100 Am² | ✅ |
| Orbit Radius | 7200 km | ✅ |
| Inclination | 35° | ✅ |
| Duration | 2 hours (7200s) | ✅ |

### Cases C & D: 3U CubeSat (Table 4.4)

**Factory Function:** `create_BC_sat()` in `common_sats.py`

| Parameter | Thesis Value | Code Match | Notes |
|-----------|--------------|------------|-------|
| Mass | 4 kg | ✅ | |
| Jxx | 0.0314 kg·m² | ✅ 0.03136... | |
| Jyy | 0.0341 kg·m² | ✅ 0.03409... | |
| Jzz | 0.0100 kg·m² | ✅ 0.01004... | |
| **Gyro Noise σ** | **0.0004 deg/s^0.5** | **0.03** | ⚠️ **SWAPPED IN THESIS** |
| **Gyro Drift σ** | **0.03 deg/s^1.5** | **0.0004** | ⚠️ **SWAPPED IN THESIS** |
| Gyro Bias Init | 0.1/√11·[1,-1,3] deg/s | ✅ | |
| MTM Noise σ | 300 nT·s^0.5 | ✅ | |
| MTM Scale | 1e4 | ✅ | |
| Sun Efficiency | 0.3 | ✅ | |
| MTQ Max | 1 Am² | ✅ | |
| MTQ Noise σ | 0.0001 Am² | ✅ | |
| Dipole Init | [0.05, 0.0001, 0.2] Am² | ✅ | |
| Orbit | 450 km, 87° | ✅ | |

**⚠️ CONFIRMED THESIS ERROR:** Table 4.4 has gyroscope noise and drift values SWAPPED. The CODE values are physically correct (noise >> drift makes physical sense).

---

## ALTRO Planner Parameters

**Source:** `dissertation_code_temp/GeneralizedADS/ADCS/src/sat_ADCS_ADCS/ADCS.py` lines 149-230

### Cost Weights
| Parameter | Value | Description |
|-----------|-------|-------------|
| angle_weight | 10 | Quaternion error weight |
| angvel_weight | 100 | Angular velocity error weight |
| angle_weight_N | 100 | Terminal quaternion weight |
| mtq_weight | 0.0001 | MTQ control effort |
| rw_weight | 0.001 | RW control effort |
| rw_AM_weight | 0.1 | RW angular momentum cost |
| rw_stic_weight | 0.01 | RW stiction avoidance |

### Solver Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| penInit | 1.0 | Initial penalty |
| penScale | 10.0 | Penalty scaling |
| regInit | 0.01 | Initial regularization |
| regScale | 1.6 | Regularization scaling |
| maxIter | 4500 | Maximum iterations |
| cmax | 0.002 | Constraint tolerance |

---

## Quick Reference: Test Configurations

### To Replicate Chapter 6 Figures
```python
# Lovera
J = np.diag([27, 17, 25])
epsilon = 0.01
kp = kv = 50
mtq_max = 50  # Am²
duration = 36000  # 10 hours
orbit = "450km, 87° inclination"

# Wisniewski
Lambda_q = 0.002 * np.eye(3)
Lambda_s = 0.003 * np.eye(3)
# Same J, mtq_max, duration, orbit
```

### To Replicate Chapter 7 Spinning Figure
```python
J = np.array([[0.1, 0, 0.00013],
              [0, 0.05, -0.00021],
              [0.00013, -0.00021, 0.005]])
disturbance = np.array([0.3e-3, 0, 0])  # Nm
q0 = np.array([-0.232, -0.664, -0.234, -0.671])
goal = "z-axis anti-ram"  # Reduced attitude
```

### To Replicate Chapter 7 Monte Carlo
```python
J = np.diag([0.005256, 0.04939, 0.04939])
q_start = np.array([0, 0, 1, 0])
q_goal = np.array([0, 1, 0, 0])
orbit = "429km, 51.5° inclination (ISS)"
trials = 100
```
