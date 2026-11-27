#!/usr/bin/env python3
"""Run the Generalized ADCS benchmark suite and write ``results.json``.

Usage (from the repository root, after ``bash benchmarks/setup.sh``)::

    python3 benchmarks/run_all.py            # full suite (~5-15 min)
    python3 benchmarks/run_all.py --quick    # short smoke test

Each tool runs in its own virtual environment under ``.bench/`` (created by
``setup.sh``), so their dependencies never conflict. This script only needs the
Python standard library. Close other heavy programs while it runs; plug in a
laptop and keep it from sleeping.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = ROOT / ".bench"
sys.path.insert(0, str(HERE))
import scenario as S  # noqa: E402

BIN = "Scripts" if os.name == "nt" else "bin"


def py(env: str) -> Path:
    return BENCH / env / BIN / ("python.exe" if os.name == "nt" else "python")


# (key, environment, runner script, extra args, scenarios)
TOOLS = [
    ("genadcs3-python", "genadcs3", "genadcs3_runner.py", ["--variant", "python"], ("coupled", "orbit")),
    ("genadcs3-cpp", "genadcs3", "genadcs3_runner.py", ["--variant", "cpp"], ("coupled", "orbit")),
    ("genadcs3-rust", "genadcs3", "genadcs3_runner.py", ["--variant", "rust"], ("coupled", "orbit")),
    ("genadcs2", "adcs2", "adcs2_runner.py", [], ("coupled", "orbit")),
    ("basilisk", "basilisk", "basilisk_runner.py", [], ("coupled", "orbit")),
    ("orekit", "orekit", "orekit_runner.py", [], ("orbit",)),
]


def run_json(cmd, timeout):
    proc = subprocess.run(cmd, cwd=HERE / "runners", capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        if os.name == "nt":
            return platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


def memory_gb():
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 ** 2, 1)
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
            return round(int(out) / 1024 ** 3, 1)
    except Exception:
        pass
    return None


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def errors(final, ref):
    out = {"position_m": math.dist(final["r"], ref["r"]),
           "velocity_m_s": math.dist(final["v"], ref["v"])}
    if final.get("dcm_bi") is not None:
        out["attitude_deg"] = S.dcm_angle_deg(final["dcm_bi"], ref["dcm_bi"])
        out["rate_rad_s"] = math.dist(final["w"], ref["w"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="short smoke test, not for publishing")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="+", help="tool keys to run (default: all available)")
    ap.add_argument("--out", default=str(HERE / "results" / "results.json"))
    args = ap.parse_args()

    fractions = (0.01, 0.02) if args.quick else S.DURATION_FRACTIONS
    headline = fractions[-1]
    tools = [t for t in TOOLS if not args.only or t[0] in args.only]
    missing = [t[0] for t in tools if not py(t[1]).exists()]
    if missing:
        print(f"Skipping (environment not set up): {', '.join(missing)}. Run benchmarks/setup.sh.")
    tools = [t for t in tools if t[0] not in missing]

    runs = []
    for scenario in ("coupled", "orbit"):
        for key, env, script, extra, scenarios in tools:
            if scenario not in scenarios:
                continue
            for fraction in fractions:
                steps = S.steps_for(fraction)
                repeats = args.repeats if fraction == headline else 1
                cmd = [str(py(env)), script, "--scenario", scenario, "--steps", str(steps),
                       "--repeats", str(repeats), *extra]
                print(f"[{scenario:7}] {key:16} {fraction:>6.3f} orbit ({steps:>6} steps) ...", end=" ", flush=True)
                try:
                    res = run_json(cmd, timeout=3600)
                except Exception as exc:  # keep going; record the failure
                    print("FAILED")
                    runs.append({"key": key, "scenario": scenario, "fraction": fraction,
                                 "steps": steps, "error": str(exc)})
                    continue
                med = sorted(res["samples_s"])[len(res["samples_s"]) // 2]
                print(f"{med:8.3f} s  ({steps / med:,.0f} steps/s)")
                res.update(key=key, fraction=fraction, median_s=med, steps_per_s=steps / med)
                runs.append(res)

    times = sorted({(r["final"] or {}).get("t", r["steps"] * S.DT)
                    for r in runs if r.get("final")})
    print("Computing high-precision reference solutions ...", flush=True)
    refs = run_json([str(py("genadcs3")), "reference_runner.py", *map(repr, times)], timeout=3600)
    for r in runs:
        if r.get("final"):
            t = r["final"].get("t", r["steps"] * S.DT)
            r["errors"] = errors(r["final"], refs[repr(float(t))])

    meta = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick,
        "repeats": args.repeats,
        "git_commit": git_commit(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_gb": memory_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "scenario": {
            "dt_s": S.DT, "orbit_period_s": S.ORBIT_PERIOD, "altitude_m": S.ALTITUDE,
            "inclination_deg": S.INCLINATION_DEG, "mass_kg": S.MASS, "inertia_kg_m2": S.INERTIA_DIAG,
            "mu": S.MU, "r_earth": S.R_EARTH, "j2": S.J2, "fractions": list(fractions),
            "descriptions": S.SCENARIOS,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "runs": runs}, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
