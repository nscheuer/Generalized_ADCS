# Paper 2 -- disturbance-robustness test (3+1 reduced)

Paired N=20, tf=1000 s. Plant carries gravity-gradient + aero drag + SRP (via a 2.1 cm cp-cg COM offset) + a residual magnetic dipole |m|=0.028 A·m². The planner plans on a CLEAN est_sat (no disturbances modeled); the TVLQR tracker must reject them.

| | clean plant | disturbed plant |
|---|---|---|
| converged (<5°) | 100% | 100% |
| mean final error | 0.11° | 0.22° |

Realized plant disturbance torque (peak / mean over one orbit):

- GG_Disturbance: 4.44e-08 / 4.14e-08 N·m
- Drag_Disturbance: 5.22e-09 / 4.67e-09 N·m
- SRP_Disturbance: 4.30e-09 / 4.30e-09 N·m
- Dipole_Disturbance: 1.14e-06 / 9.14e-07 N·m

Result: convergence is preserved under disturbances absent from the planner's model, substantiating the abstract claim. The residual magnetic dipole is the dominant term; drag/SRP are made non-zero by the cp-cg offset.
