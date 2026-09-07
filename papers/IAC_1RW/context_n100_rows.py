"""Table-3 rows for the n=100 context cells (3+0 / 3+3, PD, both tasks), in the
hand-off shape: conv5 [Wilson 95%], conv1 [Wilson 95%], median [5000-resample
bootstrap 95%], div (% final > 30 deg). Also checks seeds 0-29 against the n=30
aggregates in A_baseline_20260818_202627.json (same seeds, clamp immaterial for
these cells) and reports whether the 3+3 vs 3+1 conv5 intervals still overlap.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import error_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
CELLS = [("3MTQ+0RW", "reduced", "0rw_reduced_pd"), ("3MTQ+0RW", "full", "0rw_full_pd"),
         ("3MTQ+3RW", "reduced", "3rw_reduced_pd"), ("3MTQ+3RW", "full", "3rw_full_pd")]
WAVE = {"reduced": "wave/pd_reduced_kp1/*.pkl", "full": "wave/pd_full_kp1/*.pkl"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def boot_median(x, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = np.median(np.asarray(x)[idx], axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def finals(paths):
    fin, seeds = [], []
    for p in paths:
        with open(p, "rb") as f:
            r = pickle.load(f)
        fin.append(float(error_series(r)[-1]))
        seeds.append(int(os.path.basename(p).split("seed")[-1].split(".")[0]) if "seed" in p
                     else int(os.path.basename(p).split("_s")[-1].split(".")[0]))
    return np.asarray(fin), np.asarray(seeds)


def stats(fin):
    n = len(fin)
    k5, k1 = int(np.sum(fin <= 5)), int(np.sum(fin <= 1))
    return dict(n=n, conv5=100 * k5 / n, ci5=wilson(k5, n), conv1=100 * k1 / n, ci1=wilson(k1, n),
                med=float(np.median(fin)), cim=boot_median(fin), div=100 * float(np.mean(fin > 30)))


def fmt(name, task, s):
    return (f"| PD {name} {task} | {s['n']} | {s['conv5']:.0f} [{s['ci5'][0]:.0f},{s['ci5'][1]:.0f}] | "
            f"{s['conv1']:.0f} [{s['ci1'][0]:.0f},{s['ci1'][1]:.0f}] | "
            f"{s['med']:.2f} [{s['cim'][0]:.2f},{s['cim'][1]:.2f}] | {s['div']:.0f} |")


def main():
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260818_202627.json")))
    print("| cell | n | conv5 [CI] | conv1 [CI] | median [boot CI] | div% |")
    print("|---|---|---|---|---|---|")
    rows = {}
    for name, task, key in CELLS:
        paths = sorted(glob.glob(os.path.join(OUT, "A_trials", f"{key}_seed*.pkl")))
        if not paths:
            print(f"| PD {name} {task} | 0 | (not run) | | | |")
            continue
        fin, seeds = finals(paths)
        s = stats(fin)
        rows[key] = s
        print(fmt(name, task, s) + ("" if s["n"] == 100 else f"  <-- INCOMPLETE ({s['n']}/100)"))
    for task in ("reduced", "full"):
        fin, _ = finals(sorted(glob.glob(os.path.join(OUT, WAVE[task]))))
        s = stats(fin)
        rows[f"1rw_{task}_pd"] = s
        print(fmt("3MTQ+1RW", task, s) + "  (reference: wave cell, unchanged)")
    print()
    print("seeds 0-29 vs the n=30 aggregates (A_baseline_20260818_202627.json):")
    for name, task, key in CELLS:
        paths = sorted(glob.glob(os.path.join(OUT, "A_trials", f"{key}_seed*.pkl")))
        if not paths:
            continue
        fin, seeds = finals(paths)
        m = seeds < 30
        if m.sum() < 30:
            print(f"  {key}: only {m.sum()}/30 of seeds 0-29 on disk")
            continue
        h = d18["cells"][key]["horizons"]["5554"]
        sub = fin[m]
        print(f"  {key}: conv5 {100*np.mean(sub<=5):.1f} vs {h['conv_pct_5deg']:.1f} | "
              f"conv1 {100*np.mean(sub<=1):.1f} vs {h['conv_pct_1deg']:.1f} | "
              f"median {np.median(sub):.3f} vs {h['median_final_deg']:.3f}")
    print()
    for task in ("reduced", "full"):
        a, b = rows.get(f"3rw_{task}_pd"), rows.get(f"1rw_{task}_pd")
        if a and b:
            ov = not (a["ci5"][0] > b["ci5"][1] or b["ci5"][0] > a["ci5"][1])
            print(f"conv5 intervals 3+3 vs 3+1 ({task}): {a['ci5'][0]:.0f}-{a['ci5'][1]:.0f} vs "
                  f"{b['ci5'][0]:.0f}-{b['ci5'][1]:.0f} -> {'OVERLAP' if ov else 'SEPARATED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
