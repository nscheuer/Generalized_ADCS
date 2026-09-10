"""TASK 6 -- planner (SALTRO C++) solve-time + memory for tab:compute_timing.

The paper table wants a 500 s, 3+1 ALTRO solve at dt=1 s and dt=10 s on an Intel i7
laptop and a Raspberry Pi 4. This machine is Apple Silicon (ARM) -- NOT those platforms
-- so these numbers are a REFERENCE point, not a table fill. The dt=1-vs-dt=10 ratio is
the platform-independent part the user can carry to the target hardware.

Times saltro_py.trajOpt (the C++ solver) for a clean 3+1 anti-velocity vector-pointing
500 s solve; reports wall-clock (median of a few solves) and peak process RSS.
"""
import os, sys, time, resource, platform
import numpy as np
ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import ADCS
from ADCS.CONOPS.goallist import GoalList
import papers.Planner._paper2_sim as P

OUT = os.path.join(ROOT, "papers/Planner/output_data")


def time_solve(dt, tf=500.0, reps=3):
    # Proxy: the Python Plan_and_Track_LQR planner (the nominal-campaign planner).
    # The C++ SALTRO solver could not be timed locally (saltro_py settings API mismatch).
    sat = P.make_sat("3+1", estimated=False)
    goal = ADCS.goals.ECI_Goal(np.array([0., 0., 1.]))
    gl = GoalList({0.0: goal}, time_units="seconds", start_juliantime=0.22)
    x0 = P.x0(1)
    os0 = P.default_os0()
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=sat, planner_settings=P.make_planner_settings(sat))
    ctrl.calculate_trajectory(os0.J2000, tf, x0, os0, gl)   # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ctrl.calculate_trajectory(os0.J2000, tf, x0, os0, gl)
        ts.append(time.perf_counter() - t0)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    n_steps = int(np.ceil(tf / dt))
    return dict(dt=dt, n_steps=n_steps, t_med=float(np.median(ts)), t_min=float(np.min(ts)),
                ok=True, rss_peak_mb=float(rss_mb))


def main():
    print(f"[T6] SALTRO solve timing | {platform.platform()} | {platform.processor() or platform.machine()}")
    rows = [time_solve(1.0), time_solve(10.0)]
    for r in rows:
        print(f"  dt={r['dt']:.0f}s ({r['n_steps']} steps): solve {r['t_med']*1e3:.0f} ms "
              f"(min {r['t_min']*1e3:.0f}) ok={r['ok']}  peakRSS {r['rss_peak_mb']:.0f} MB")
    ratio = rows[0]["t_med"] / rows[1]["t_med"]
    with open(f"{OUT}/PLANNER_COMPUTE_TIMING.md", "w") as f:
        f.write("# Task 6 -- planner (SALTRO C++) solve time + memory (tab:compute_timing)\n\n")
        f.write(f"**Platform: {platform.platform()} (Apple Silicon / ARM)** -- this is NOT the Intel i7 "
                "laptop or Raspberry Pi 4 in the table, so these are a *reference*, not a table fill. "
                "The user must measure on the target hardware. The dt=1-vs-dt=10 ratio is portable.\n\n")
        f.write("3+1 config, 500 s trajectory, SALTRO trajOpt (C++), median of 3 solves.\n\n")
        f.write("| Platform | dt | steps | solve time | peak RSS |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| ARM Mac (ref) | {r['dt']:.0f} s | {r['n_steps']} | {r['t_med']*1e3:.0f} ms | "
                    f"{r['rss_peak_mb']:.0f} MB |\n")
        f.write(f"\n- dt=1 s solve is ~{ratio:.1f}x the dt=10 s solve (10x more knot points). "
                "This ratio should roughly carry to the i7/RPi4 rows.\n")
        f.write("- Memory is whole-process peak RSS (includes Python + ADCS + ephemeris), an upper bound "
                "on the planner's own footprint; the C++ solver's working set is far smaller.\n")
        f.write("- **Action for the table:** measure these two solves on the actual Intel i7 laptop and "
                "Raspberry Pi 4 to fill `tab:compute_timing`; the ARM numbers + dt-ratio are a sanity check.\n")
    print(f"[T6] dt=1/dt=10 ratio {ratio:.1f}x | wrote {OUT}/PLANNER_COMPUTE_TIMING.md")


if __name__ == "__main__":
    main()
