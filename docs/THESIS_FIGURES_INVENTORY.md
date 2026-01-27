# Complete Thesis Figure Inventory

## Schematics / Diagrams (Manual - Not Generated)
These are conceptual diagrams that can't be auto-generated from simulation:

| Figure | Description | Type |
|--------|-------------|------|
| `ALTRO.png` | ALTRO algorithm structure | Diagram |
| `RW_AM.png` | RW angular momentum cost | Concept |
| `sat_diagram.png` | Satellite class structure | Diagram |
| `quaternion_set.png` | Goal type visualization | Diagram |
| `problemsolving.png` | Problem solving approach | Diagram |
| `beavercube.png` | BeaverCube photo | Photo |
| `imtq_back`, `imtq_front` | Hardware photos | Photo |
| `sunpointing_thesis.png` | Sun pointing concept | Diagram |
| `venn.png` | Venn diagram | Diagram |
| `single axis.png` | Single axis concept | Diagram |
| `anim_stack.png` | Trajectory visualization (3D render) | Special |

---

## Data Figures - Chapter 4: Estimation

### Case A: TRMM with Initial Attitude and Bias Errors
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `log_angular_error_TRMM_initially_off.png` | Angular error log scale | ✅ |
| `log_norm_av_TRMM_initially_off.png` | Angular velocity error | ✅ |
| `usque_1mrp_TRMM_initially_off_3sig.png` | USQUE 1s MRP with bounds | ✅ |
| `usque_10mrp_TRMM_initially_off_3sig.png` | USQUE 10s MRP with bounds | ✅ |
| `mine_1mrp_TRMM_initially_off_3sig.png` | DAF 1s MRP with bounds | ✅ |
| `mine_10mrp_TRMM_initially_off_3sig.png` | DAF 10s MRP with bounds | ✅ |

### Case B: TRMM with Small Initial Bias Errors Only
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `log_angular_error_TRMM_initially_close.png` | Angular error log scale | ✅ |
| `log_norm_av_TRMM_initially_close.png` | Angular velocity error | ✅ |
| `usque_1mrp_TRMM_initially_close_3sig.png` | USQUE 1s with bounds | ✅ |
| `usque_10mrp_TRMM_initially_close_3sig.png` | USQUE 10s with bounds | ✅ |
| `mine_1mrp_TRMM_initially_close_3sig.png` | DAF 1s with bounds | ✅ |
| `mine_10mrp_TRMM_initially_close_3sig.png` | DAF 10s with bounds | ✅ |

### Case C: CubeSat with Initial Attitude and Bias Errors
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `log_angular_error_BC_initially_off.png` | Angular error | ✅ |
| `log_norm_av_BC_initially_off.png` | Angular velocity error | ✅ |
| `mrp_BC_initially_off_3sig.png` | MRP with bounds | ✅ |
| `axes_av_BC_initially_off_3sig.png` | Angular velocity with bounds | ✅ |

### Case D: CubeSat with Small Initial Bias Errors Only
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `log_angular_error_BC_initially_close.png` | Angular error | ✅ |
| `log_norm_av_BC_initially_close.png` | Angular velocity error | ✅ |
| `mrp_BC_initially_close_3sig.png` | MRP with bounds | ✅ |
| `axes_av_BC_initially_close_3sig.png` | Angular velocity with bounds | ✅ |

### Case E-G: CubeSat Inclusion Tests
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `log_angular_error_BC_abias_inclusion.png` | Actuator bias inclusion | ✅ |
| `log_angular_error_BC_dist_inclusion.png` | Disturbance inclusion | ✅ |
| `log_angular_error_BC_prop_inclusion.png` | Propagation torque inclusion | ✅ |
| `prop_torque_BC_prop_inclusion.png` | Propagation torque tracking | ✅ |

### Case G: Many Variables (Full Estimation)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `many_var/log_angular_error_caseg.png` | Angular error | ✅ |
| `many_var/angular_error_caseg.png` | Angular error linear | ✅ |
| `many_var/axes_av_caseg.png` | Angular velocity | ✅ |
| `many_var/axes_av_caseg_3sig.png` | Angular velocity with bounds | ✅ |
| `many_var/mrp_caseg_3sig.png` | MRP with bounds | ✅ |
| `many_var/am_caseg.png` | Angular momentum | ✅ |
| `many_var/am_caseg_3sig.png` | Angular momentum with bounds | ✅ |
| `many_var/gb_caseg.png` | Gyro bias | ✅ |
| `many_var/gyrobias_caseg_3sig.png` | Gyro bias with bounds | ✅ |
| `many_var/mb_caseg.png` | MTM bias | ✅ |
| `many_var/mtmbias_caseg_3sig.png` | MTM bias with bounds | ✅ |
| `many_var/sb_caseg.png` | Sun sensor bias | ✅ |
| `many_var/sunbias_caseg_3sig.png` | Sun sensor bias with bounds | ✅ |
| `many_var/dipole_caseg.png` | Magnetic dipole | ✅ |
| `many_var/dipole_caseg_3sig.png` | Magnetic dipole with bounds | ✅ |
| `many_var/dipole_nobb_caseg.png` | Dipole no B-bias | ✅ |
| `many_var/proptorq_caseg.png` | Propagation torque | ✅ |
| `many_var/proptorq_caseg_3sig.png` | Propagation torque with bounds | ✅ |
| `many_var/abias_caseg.png` | Actuator bias | ✅ |

**Estimation Total: 41 data figures**

---

## Data Figures - Chapter 6: Disturbance Control

### Wie Comparison (3RW Large Satellite)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_wie.png` | Angular error | ✅ |
| `log_angular_error_wie.png` | Angular error log | ✅ |
| `axes_av_wie.png` | Angular velocity | ✅ |
| `ctrl_wie.png` | Control effort | ✅ |
| `rpy_wie_limited.png` | RPY error | ✅ |

### Lovera Comparison (MTQ-only)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_lovera.png` | Angular error | ✅ |
| `log_angular_error_lovera.png` | Angular error log | ✅ |
| `ctrl_lovera.png` | Control effort | ✅ |
| `axes_av_lovera.png` | Angular velocity | ✅ |
| `rpy_lovera_limited.png` | RPY error | ✅ |

### Lovera CubeSat
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_lovera_CubeSat.png` | Angular error | ✅ |
| `log_angular_error_lovera_CubeSat.png` | Angular error log | ✅ |
| `ctrl_lovera_CubeSat.png` | Control effort | ✅ |
| `axes_av_lovera_CubeSat.png` | Angular velocity | ✅ |
| `rpy_lovera_CubeSat.png` | RPY error | ✅ |

### Wisniewski Comparison (MTQ Sliding Mode)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_wisniewski.png` | Angular error | ✅ |
| `log_angular_error_wisniewski.png` | Angular error log | ✅ |
| `ctrl_wisniewski.png` | Control effort | ✅ |
| `axes_av_wisniewski.png` | Angular velocity | ✅ |
| `rpy_wisniewski_limited.png` | RPY error | ✅ |

### Wisniewski 10s Timestep
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_wisniewski10.png` | Angular error | ✅ |
| `log_angular_error_wisniewski10.png` | Angular error log | ✅ |
| `ctrl_wisniewski10.png` | Control effort | ✅ |
| `axes_av_wisniewski10.png` | Angular velocity | ✅ |
| `rpy_wisniewski10_limited.png` | RPY error | ✅ |

### Wisniewski CubeSat
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_wisniewski_CubeSat.png` | Angular error | ✅ |
| `log_angular_error_wisniewski_CubeSat.png` | Angular error log | ✅ |
| `ctrl_wisniewski_CubeSat.png` | Control effort | ✅ |
| `axes_av_wisniewski_CubeSat.png` | Angular velocity | ✅ |
| `rpy_wisniewski_CubeSat.png` | RPY error | ✅ |

### Wisniewski Twist Variants
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_wisniewski_twist.png` | Angular error | ✅ |
| `log_angular_error_wisniewski_twist.png` | Angular error log | ✅ |
| `ctrl_wisniewski_twist.png` | Control effort | ✅ |
| `axes_av_wisniewski_twist.png` | Angular velocity | ✅ |
| `rpy_wisniewski_twist_limited.png` | RPY error | ✅ |
| `angular_error_wisniewski_twist_CubeSat.png` | CubeSat angular error | ✅ |
| `log_angular_error_wisniewski_twist_CubeSat.png` | CubeSat log error | ✅ |
| `ctrl_wisniewski_twist_CubeSat.png` | CubeSat control | ✅ |
| `axes_av_wisniewski_twist_CubeSat.png` | CubeSat angular velocity | ✅ |
| `rpy_wisniewski_twist_CubeSat.png` | CubeSat RPY | ✅ |

### Disturbed Comparisons
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error_disturbed_wis_twist_comp.png` | Disturbed comparison | ✅ |
| `axes_av_disturbed_wis_twist_comp.png` | Angular velocity | ✅ |
| `ctrl_disturbed_wis_twist_comp.png` | Control effort | ✅ |
| `rpy_disturbed_wis_twist_comp.png` | RPY | ✅ |
| `angular_error_disturbed_w_control_wis_twist_comp.png` | With control | ✅ |
| `axes_av_disturbed_w_control_wis_twist_comp.png` | With control AV | ✅ |
| `ctrl_disturbed_w_control_wis_twist_comp.png` | With control effort | ✅ |
| `rpy_disturbed_w_control_wis_twist_comp.png` | With control RPY | ✅ |
| `angular_error_disturbed_alt_wis_twist_comp.png` | Alt comparison | ✅ |
| `axes_av_disturbed_alt_wis_twist_comp.png` | Alt AV | ✅ |
| `ctrl_disturbed_alt_wis_twist_comp.png` | Alt control | ✅ |
| `rpy_disturbed_alt_wis_twist_comp.png` | Alt RPY | ✅ |
| `angular_error_disturbed_w_control_alt_wis_twist_comp.png` | Alt w/ control | ✅ |
| `axes_av_disturbed_w_control_alt_wis_twist_comp.png` | Alt w/ control AV | ✅ |
| `ctrl_disturbed_w_control_alt_wis_twist_comp.png` | Alt w/ control effort | ✅ |
| `rpy_disturbed_w_control_alt_wis_twist_comp.png` | Alt w/ control RPY | ✅ |

### CubeSat Disturbed
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `angular_error__cubesat_wis_twist_comp.png` | CubeSat comparison | ✅ |
| `axes_av__cubesat_wis_twist_comp.png` | CubeSat AV | ✅ |
| `ctrl__cubesat_wis_twist_comp.png` | CubeSat control | ✅ |
| `rpy__cubesat_wis_twist_comp.png` | CubeSat RPY | ✅ |
| `angular_error__disturbed_cubesat_wis_twist_comp.png` | Disturbed CubeSat | ✅ |
| `axes_av__disturbed_cubesat_wis_twist_comp.png` | Disturbed CubeSat AV | ✅ |
| `ctrl__disturbed_cubesat_wis_twist_comp.png` | Disturbed CubeSat ctrl | ✅ |
| `rpy__disturbed_cubesat_wis_twist_comp.png` | Disturbed CubeSat RPY | ✅ |

**Disturbance Control Total: ~70 data figures**

---

## Data Figures - Chapter 7: Planning

### Spinning Solution
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `spinning_ang.png` | Pointing error | ✅ |
| `spinning_av.png` | Angular velocity | ✅ |
| `spinning_cmd.png` | Control commands | ✅ |

### Simple Slew MC (Full Attitude)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `simple_slew/mtq_montecarlo.png` | MTQ histogram | ✅ |
| `simple_slew/mtq_montecarlo_traj.png` | MTQ trajectories | ✅ |
| `simple_slew/mtq_good_quaternion.png` | MTQ good example | ✅ |
| `simple_slew/mtq_bad_quaternion.png` | MTQ bad example | ✅ |
| `simple_slew/1W_montecarlo.png` | 3+1 histogram | ✅ |
| `simple_slew/1W_montecarlo_traj.png` | 3+1 trajectories | ✅ |
| `simple_slew/1W_mom_montecarlo.png` | 3+1 momentum | ✅ |
| `simple_slew/1W_good_quaternion.png` | 3+1 good example | ✅ |
| `simple_slew/1W_bad_quaternion.png` | 3+1 bad example | ✅ |

### Single Target Imaging MC (Reduced Attitude)
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `single_target_imaging/mtq_quatset_montecarlo.png` | MTQ histogram | ✅ |
| `single_target_imaging/mtq_quatset_montecarlo_traj.png` | MTQ trajectories | ✅ |
| `single_target_imaging/mtq_quatset_good_quaternion.png` | MTQ good | ✅ |
| `single_target_imaging/mtq_quatset_bad_quaternion.png` | MTQ bad | ✅ |
| `single_target_imaging/1W_quatset_montecarlo.png` | 3+1 histogram | ✅ |
| `single_target_imaging/1W_quatset_montecarlo_traj.png` | 3+1 trajectories | ✅ |
| `single_target_imaging/1W_quatset_mom_montecarlo.png` | 3+1 momentum | ✅ |
| `single_target_imaging/1W_quatset_good_quaternion.png` | 3+1 good | ✅ |
| `single_target_imaging/1W_quatset_bad_quaternion.png` | 3+1 bad | ✅ |

### Multi-Target Imaging MC
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `multi_target_imaging/mtq_multi_2_montecarlo.png` | MTQ histogram | ✅ |
| `multi_target_imaging/mtq_multi_montecarlo_traj.png` | MTQ trajectories | ✅ |
| `multi_target_imaging/mtq_multi_good_quaternion.png` | MTQ good | ✅ |
| `multi_target_imaging/mtq_multi_bad_quaternion.png` | MTQ bad | ✅ |
| `multi_target_imaging/1W_multi_2_montecarlo.png` | 3+1 histogram | ✅ |
| `multi_target_imaging/1W_multi_montecarlo_traj.png` | 3+1 trajectories | ✅ |
| `multi_target_imaging/1W_multi_good_quaternion.png` | 3+1 good | ✅ |
| `multi_target_imaging/1W_multi_bad_quaternion.png` | 3+1 bad | ✅ |

### Sequential Planning
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `sequential/plan_quat_plot.png` | Quaternion | ✅ |
| `sequential/plan_av_plot.png` | Angular velocity | ✅ |
| `sequential/planvecang.png` | Pointing error | ✅ |
| `sequential/planctrl_plot.png` | Control commands | ✅ |
| `sequential/_logplanvecang.png` | Log pointing error | ✅ |

### Trajectory Following
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `follow_quat.png` | Following quaternion | ✅ |
| `follow_av.png` | Following angular velocity | ✅ |
| `follow_ang.png` | Following pointing error | ✅ |
| `follow_cmd.png` | Following commands | ✅ |

### Two-Goal Trajectory
| Figure | Description | Can Generate |
|--------|-------------|--------------|
| `anim_plots.png` | Trajectory plots | ✅ |

**Planning Total: ~40 data figures**

---

## Summary

| Chapter | Data Figures | Schematics |
|---------|--------------|------------|
| Estimation (Ch 4) | 41 | 1 |
| Disturbance (Ch 6) | ~70 | 0 |
| Planning (Ch 7) | ~40 | 5 |
| **Total** | **~151** | **~11** |

All ~151 data figures CAN be generated from simulation!
