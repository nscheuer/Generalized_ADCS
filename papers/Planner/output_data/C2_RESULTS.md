# C2 — disturbance-assisted desaturation: feasibility + honest negative

**Outcome: NOT demonstrated.** Two distinct reasons, both characterized below. Honest negative
(per the brief's "don't over-invest / report effort"), with a clear path if pursued.

## 1. Gravity-gradient / aero are physically too weak (verified)
Momentum a disturbance can dump over one ~5800 s orbit vs. h_max = 3.6 mN·m·s:

| disturbance | torque | dump/orbit | % of h_max |
|---|---|---|---|
| gravity-gradient | ~4.3e-8 N·m | 0.25 mN·m·s | **6.9%** |
| drag/SRP (cp–cg) | ~5e-9 N·m | 0.03 mN·m·s | 0.8% |
| residual dipole 0.05 A·m² | ~1.5e-6 N·m | 8.7 mN·m·s | 242% |
| residual dipole 0.10 A·m² | ~3.0e-6 N·m | 17 mN·m·s | 483% |
| magnetorquer 0.2 A·m² | ~6e-6 N·m | 35 mN·m·s | 967% |

GG/aero would need ~14+ orbits to desaturate once — **not** a single-trajectory mechanism, even
with the spin-demo's large inertia asymmetry (~25%/orbit). The GG-assisted version the brief asked
for is physically infeasible. The only environmental torque strong enough is a **magnetic** one.

## 2. The strong-dipole demo DIVERGED on the OldPlanner
Designed satellite: a modeled residual dipole of 0.2 A·m² (≈ magnetorquer strength), wheel started
near saturation (h0 = 0.9 h_max = 3.24 mN·m·s), reduced-attitude pointing (roll free). Result
(`fig_c2_dipole_desat.png`):
- **wheel momentum ran away: 2.88 → 63 mN·m·s** (17× h_max), and
- **pointing was lost: 80° final** (never acquired; diverged to ~170°).

Root cause: the OldPlanner cost (`_paper2_sim.make_planner_settings`) has **no
momentum-minimization term** — only a saturation *constraint*, which here did not drive active
desaturation — and a disturbance *stronger than the actuators* destabilized the closed loop. So this
configuration can neither hold pointing nor desaturate.

## Path forward (effort HIGH — deferred)
A real disturbance-assisted desat demo needs:
1. the **SALTRO planner with an explicit momentum weight** (`rw_AM_weight`, which drives wheel
   momentum down — cf. the P2.6 leak work), not the OldPlanner;
2. a **moderate** steerable disturbance (dipole ~0.03–0.05 A·m² — below actuator strength so the loop
   stays stable, but enough to dump several mN·m·s/orbit);
3. a momentum target in the cost so the optimizer actively offloads.
Essentially the spin-demo machinery (SALTRO trajOpt) repurposed for desat — a substantial new build,
not attempted here.

**Net:** the abstract's "exploit external disturbances to desaturate" is *physically* supportable only
by a magnetic dipole (GG/aero too weak), and a working demo needs the momentum-aware SALTRO planner;
the OldPlanner cannot show it. C1 (foresight pre-positioning) DID land cleanly — see `C1_RESULTS.md`.
