# Dissertation Figure to Code Mapping

## Source Code Repository
**GitHub:** `patrickmckeen/PhD_Dissertation_Code`
- **GeneralizedADS/** - Python-based estimation, disturbance, and control code
- **beavercube-adcs-master_thesis/** - MATLAB-based trajectory planning code

## Chapter 4: Estimation (estimation.tex)

### Case A & B: TRMM Satellite (Large satellite with magnetometers & sun sensors)
**Test File:** `GeneralizedADS/ADCS/thesis_estimation_tests_rwmtq.py` (and `thesis_better_estimation_tests_rwmtq.py`)
**Plotting File:** `GeneralizedADS/ADCS/thesis_est_case_g_plots.py`
**Data Folder:** `GeneralizedADS/ADCS/thesis_test_files/est_files/`

| Figure | Image File | Data Source | Description |
|--------|-----------|-------------|-------------|
| Case A Angular Error | `log_angular_error_TRMM_initially_off.png` | `*_TRMM_initially_off*` | TRMM with initial attitude/bias errors |
| Case A AV Error | `log_norm_av_TRMM_initially_off.png` | Same | Angular velocity error |
| Case A 3σ Bounds | `usque_*_TRMM_initially_off_3sig.png`, `mine_*_TRMM_initially_off_3sig.png` | Same | MRP error with bounds |
| Case B Angular Error | `log_angular_error_TRMM_initially_close.png` | `*_TRMM_initially_close*` | TRMM with small initial errors |
| Case B AV Error | `log_norm_av_TRMM_initially_close.png` | Same | |
| Case B 3σ Bounds | `usque_*_TRMM_initially_close_3sig.png`, `mine_*_TRMM_initially_close_3sig.png` | Same | |

**Configuration (from test file):**
- Orbit: `orb_file = "lovera_orb_1"` 
- Filter comparison: Dynamics-Aware Filter vs USQUE (Crassidis)
- Time steps: 1s and 10s
- Sensors: MTM, Sun sensors, Gyros (with biases)

### Case C & D: 3U CubeSat (Lower-quality sensors)
**Test File:** Same as above
**Data Folder:** `GeneralizedADS/ADCS/thesis_test_files/est_files/`

| Figure | Image File | Data Source | Description |
|--------|-----------|-------------|-------------|
| Case C Angular Error | `log_angular_error_BC_initially_off.png` | `thesis_6U_quat_RWMTQ_*_initial_off*` | CubeSat with initial errors |
| Case C AV Error | `log_norm_av_BC_initially_off.png` | Same | |
| Case C 3σ | `mrp_BC_initially_off_3sig.png`, `axes_av_BC_initially_off_3sig.png` | Same | |
| Case D Angular Error | `log_angular_error_BC_initially_close.png` | `thesis_6U_quat_RWMTQ_*` (not initial_off) | CubeSat with small initial errors |
| Case D AV Error | `log_norm_av_BC_initially_close.png` | Same | |
| Case D 3σ | `mrp_BC_initially_close_3sig.png`, `axes_av_BC_initially_close_3sig.png` | Same | |

**Configuration:**
- Satellite: `create_GPS_6U_sat()` function
- Sensors: MTM (scale=1e0), Sun sensors, MEMS Gyros (poor quality)
- Actuators: 3 MTQ + 3 RW
- Control mode: `GovernorMode.MTQ_W_RW_PD`

### Case E & F: Disturbance/Bias Inclusion Tests
| Figure | Image File | Description |
|--------|-----------|-------------|
| Bias Inclusion | `log_angular_error_BC_abias_inclusion.png` | Effect of ignoring actuator bias |
| Disturbance Inclusion | `log_angular_error_BC_dist_inclusion.png` | Effect of ignoring disturbances |
| Prop Inclusion | `log_angular_error_BC_prop_inclusion.png` | Effect of propulsion torque |
| Prop Torque Est | `prop_torque_BC_prop_inclusion.png` | Propulsion torque tracking |

### Case G: Many Variables (6U CubeSat with full estimation)
**Test File:** `thesis_better_estimation_tests_rwmtq.py`
**Plotting File:** `thesis_est_case_g_plots.py`
**Data:** `thesis_6U_quat_RWMTQ_betterest_w_all_20240819*`

| Figure | Image File | Description |
|--------|-----------|-------------|
| Angular Error | `many_var/angular_error_caseg.png` | Full estimation angular error |
| AV Error | `many_var/axes_av_caseg.png` | Angular velocity error |
| Stored AM | `many_var/am_caseg.png` | Reaction wheel momentum tracking |
| Prop Torque | `many_var/proptorq_caseg.png` | Propulsion torque estimation |
| Dipole | `many_var/dipole_caseg.png` | Residual dipole tracking |
| Dipole (no BB) | `many_var/dipole_nobb_caseg.png` | Dipole without B-dot |
| Gyro Bias | `many_var/gb_caseg.png` | Gyroscope bias estimation |
| MTM Bias | `many_var/mb_caseg.png` | Magnetometer bias estimation |
| Sun Bias | `many_var/sb_caseg.png` | Sun sensor bias estimation |
| 3σ bounds | `many_var/*_3sig.png` | Various 3-sigma bound plots |

**Configuration:**
```python
main_sat_rwmtq = create_GPS_6U_sat_betterest(
    real=True, rand=False,
    mtm_scale=1e0,
    use_dipole=True,
    use_mtq=True,
    include_mtmbias=True,
    include_mtqbias=False,
    include_gbias=True,
    include_sbias=True,
    use_prop=True
)
```

---

## Chapter 5: Disturbance Rejection (disturbance.tex)

### Control Law Comparisons
**Test File:** `GeneralizedADS/ADCS/thesis_disturbance_tests_rwmtq.py`

| Figure Set | Image Pattern | Controller | Satellite |
|-----------|---------------|------------|-----------|
| Wie | `*_wie.png` | Wie PD | Large satellite |
| Lovera | `*_lovera.png` | Lovera/Silani | Large satellite |
| Wisniewski | `*_wisniewski.png` | Wisniewski | Large satellite |
| Wisniewski Twist | `*_wisniewski_twist.png` | Wisniewski w/ twist comp | Large satellite |
| Wisniewski CubeSat | `*_wisniewski_CubeSat.png` | Wisniewski | CubeSat |
| Lovera CubeSat | `*_lovera_CubeSat.png` | Lovera/Silani | CubeSat |
| Wisniewski Twist CubeSat | `*_wisniewski_twist_CubeSat.png` | Wisniewski w/ twist | CubeSat |

**Figure Types (prefix):**
- `angular_error_` - Pointing error over time
- `log_angular_error_` - Log scale pointing error
- `axes_av_` - Angular velocity by axis
- `ctrl_` - Control effort
- `rpy_` - Roll/pitch/yaw (commented out in most cases)

### Disturbance Compensation Tests
| Figure | Image Pattern | Description |
|--------|---------------|-------------|
| With/Without Control | `*_disturbed_w_control_wis_twist_comp.png` | Disturbance w/ feedforward |
| Disturbed (no comp) | `*_disturbed_wis_twist_comp.png` | Disturbance w/o feedforward |
| CubeSat disturbed | `*_disturbed_cubesat_wis_twist_comp.png` | CubeSat with disturbances |
| Generalized control | `*_gen_cubesat_wis_twist_comp.png` | Generalized disturbance tracking |

---

## Chapter 7: Trajectory Planning (planning.tex)

### ALTRO/iLQR Planning
**Code Location:** 
- MATLAB: `beavercube-adcs-master_thesis/ALiLQR_BC_PM.m`, `call_simulation.m`
- Python: `GeneralizedADS/ADCS/trajectory_planner/`
- Python plot: `GeneralizedADS/thesis_spindist_plan_plots.py`

| Figure | Image File | Description |
|--------|-----------|-------------|
| ALTRO Overview | `ALTRO.png` | Algorithm diagram |
| RW AM Management | `RW_AM.png` | Reaction wheel momentum |
| Animation Stack | `anim_stack.png` | Trajectory visualization |
| Animation Plots | `anim_plots.png` | Trajectory plots |
| Spinning Angle | `spinning_ang.png` | Spinning maneuver angle |
| Spinning AV | `spinning_av.png` | Spinning maneuver velocity |
| Spinning Cmd | `spinning_cmd.png` | Spinning maneuver commands |

### Monte Carlo Planning Results
**Data Location:** `GeneralizedADS/ADCS/thesis_test_files/` (various `*_MTQ_*_LQR_*` folders)

#### Simple Slew (images/simple_slew/)
| Figure | Configuration | Description |
|--------|---------------|-------------|
| `mtq_montecarlo.png` | MTQ only | MC pointing error |
| `mtq_montecarlo_traj.png` | MTQ only | MC trajectories |
| `mtq_good_quaternion.png` | MTQ only | Good trajectory example |
| `mtq_bad_quaternion.png` | MTQ only | Bad trajectory example |
| `1W_montecarlo.png` | 1RW + MTQ | MC pointing error |
| `1W_montecarlo_traj.png` | 1RW + MTQ | MC trajectories |
| `1W_mom_montecarlo.png` | 1RW + MTQ | RW momentum during MC |
| `1W_good_quaternion.png` | 1RW + MTQ | Good trajectory example |
| `1W_bad_quaternion.png` | 1RW + MTQ | Bad trajectory example |

#### Single Target Imaging (images/single_target_imaging/)
Same pattern as simple_slew with `quatset_` prefix:
- `mtq_quatset_*.png` - MTQ with quaternion set goals
- `1W_quatset_*.png` - 1RW+MTQ with quaternion set goals

#### Multi-Target Imaging (images/multi_target_imaging/)
Same pattern with `multi_` prefix:
- `mtq_multi_*.png` - MTQ multi-target
- `1W_multi_*.png` - 1RW+MTQ multi-target

#### Sequential Planning (images/sequential/)
**Plot File:** `GeneralizedADS/ADCS/thesis_seq_plots.py`
**Data:** `GeneralizedADS/ADCS/thesis_test_files/` (LQR folders)

| Figure | Image File | Description |
|--------|-----------|-------------|
| Planned Quaternion | `plan_quat_plot.png` | Quaternion trajectory |
| Planned AV | `plan_av_plot.png` | Angular velocity plan |
| Pointing Angle | `planvecang.png` | Vector angle to target |
| Control Commands | `planctrl_plot.png` | Planned controls |
| Log Pointing | `_logplanvecang.png` | Log-scale pointing error |

---

## Satellite Factory Functions

**Location:** `GeneralizedADS/satellite_hardware/src/sat_ADCS_satellite/satellite.py`

Key satellite creation functions:
- `create_GPS_6U_sat()` - Standard 6U CubeSat
- `create_GPS_6U_sat_betterest()` - 6U with improved estimation params

**Parameters:**
```python
create_GPS_6U_sat(
    real=True/False,      # Real vs estimated satellite
    rand=False,           # Randomization
    mtm_scale=1e0,        # MTM noise scaling
    use_dipole=True,      # Include residual dipole
    use_mtq=True,         # Include magnetorquers
    include_mtmbias=True, # MTM bias
    include_mtqbias=False,# MTQ bias
    include_gbias=True,   # Gyro bias
    include_sbias=True,   # Sun sensor bias
    use_prop=True,        # Propulsion disturbance
    care_about_eclipse=False # Eclipse handling
)
```

---

## Key Test Configuration Parameters

From `thesis_better_estimation_tests_rwmtq.py`:

```python
# Initial state
q0 = normalize(np.array([0.153,0.685,0.695,0.153]))
w0 = 2*np.array([0.01,0.01,0.001])  # rad/s
h0 = -0.001*(15/1000)*np.ones(3)    # Nms

# Goals system
quat_rwmtq_goals = Goals(
    {
        0.2: GovernorMode.NO_CONTROL,
        0.22+5*sec2cent: GovernorMode.RWBDOT_WITH_EKF,
        0.22+20*sec2cent: GovernorMode.MTQ_W_RW_PD_MINE
    },
    {
        0.2: (PointingGoalVectorMode.PROVIDED_MRP_WISNIEWSKI, ...),
        ...
    },
    {0.22: -unitvecs[0], ...}  # Body axes
)

# Orbit file
orb_file = "lovera_orb_1"
```

---

## Notes on Reproducing Results

1. **Estimation Results:** Run `thesis_better_estimation_tests_rwmtq.py` with appropriate test cases uncommented
2. **Estimation Plots:** Run `thesis_est_case_g_plots.py` after tests complete
3. **Planning Results:** MATLAB code in `beavercube-adcs-master_thesis/` or Python in `trajectory_planner/`
4. **Disturbance Tests:** Run `thesis_disturbance_tests_rwmtq.py`

Data is typically saved to `thesis_test_files/` or `thesis_test_files/est_files/` with timestamps.
