"""P1.4 -- MC validation digest (Paper 1 master reference table).

Derived (no compute): harvests the per-experiment CSV summaries into one
master table consolidating Paper 1's MC validation:
  * P1.1  LP1 vs LP2 paired (allocation)  -> tab_lp1_vs_lp2.csv
  * P1.2  same PD across configs          -> tab_same_pd.csv
  * P1.3  four laws (full-attitude)        -> tab_difflaw_mc.csv

Honesty: the three experiments do NOT share one testbed. P1.1 and P1.2 use the
canonical PD gains (kp=5e-5, kd=1e-3) and a common saturation metric (1-abar),
so those rows are mutually consistent on 1-abar; absolute convergence on that
harsh testbed is reported but flagged "relative". P1.3 uses per-law gains on a
full-attitude task and is reported on its own basis. Each row carries a
``basis`` tag; paired vs absolute is labelled per row.

Emits: output_data/P1.4_digest.{tex,csv}, P1.4_RESULTS.md (separate).
"""

import os
import csv

OUT = "papers/Generalized_ACS/output_data"


def read_csv(name):
    with open(f"{OUT}/{name}") as f:
        return list(csv.DictReader(f))


def f(x, d="-"):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return d


def main():
    rows = []

    # --- P1.1 LP1 vs LP2 (allocation): LP rows, representative h levels ---
    lp = read_csv("tab_lp1_vs_lp2.csv")
    for r in lp:
        if r["allocator"] == "LP" and r["h_frac_of_hmax"] in ("0.0000", "0.9000"):
            rows.append({
                "exp": "P1.1 alloc", "config": r["config"],
                "controller": "LP-PD", "alloc": "LP", "task": "vector",
                "n": "100", "conv": f(r["conv_pct"]) + "%",
                "mean": f(r["mean_err_deg"]), "p95": f(r["p95_err_deg"]),
                "sat": f(r["gyro_shortfall_proxy"]),
                "basis": "absolute (harsh testbed; cite paired Δ, 1-ᾱ)",
            })

    # --- P1.2 same PD across configs ---
    sp = read_csv("tab_same_pd.csv")
    for r in sp:
        rows.append({
            "exp": "P1.2 same-PD", "config": r["config"],
            "controller": "LP-PD", "alloc": "LP", "task": r["task"],
            "n": r["n"], "conv": f(r["conv_pct"]) + "%",
            "mean": f(r["mean_final_deg"]), "p95": f(r["p95_final_deg"]),
            "sat": f(r["mean_saturation"]),
            "basis": f"absolute (rel.); {r['controllability']}",
        })

    # --- P1.3 four laws (full-attitude) ---
    dl = read_csv("tab_difflaw_mc.csv")
    for r in dl:
        rows.append({
            "exp": "P1.3 laws", "config": "3MTQ+1RW",
            "controller": r["law"], "alloc": "LP", "task": r["task"],
            "n": r["n"], "conv": f(r["conv_pct"]) + "%",
            "mean": f(r["mean_final_deg"]), "p95": f(r["p95_final_deg"]),
            "sat": "-",
            "basis": f"absolute; Δ vs LP-PD {f(r['paired_delta_vs_LPPD_mean_deg'])}",
        })

    cols = ["exp", "config", "controller", "alloc", "task", "n", "conv",
            "mean", "p95", "sat", "basis"]
    hdr = ["Experiment", "Config", "Controller", "Alloc", "Task", "n",
           "Conv%", "Mean(deg)", "P95(deg)", "1-abar", "Basis"]

    with open(f"{OUT}/P1.4_digest.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[c] for c in cols])

    with open(f"{OUT}/P1.4_digest.tex", "w") as fp:
        fp.write("% Paper 1 MC validation digest. Testbeds differ across\n")
        fp.write("% experiments; 1-abar (saturation) is the cross-consistent,\n")
        fp.write("% gain-robust metric. Absolute conv% on the harsh P1.1/P1.2\n")
        fp.write("% testbed is relative, not comparable to P2.1.\n")
        fp.write("\\begin{tabular}{l l l l l r r r r r}\n\\toprule\n")
        fp.write(" & ".join(hdr[:-1]) + " \\\\\n\\midrule\n")
        last = None
        for r in rows:
            if last is not None and r["exp"] != last:
                fp.write("\\addlinespace\n")
            last = r["exp"]
            fp.write(" & ".join([r["exp"], r["config"], r["controller"],
                                 r["alloc"], r["task"], r["n"], r["conv"],
                                 r["mean"], r["p95"], r["sat"]]) + " \\\\\n")
        fp.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[P1.4] wrote {OUT}/P1.4_digest.{{tex,csv}} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['exp']:13s} {r['config']:9s} {r['controller']:11s} "
              f"{r['task']:13s} conv {r['conv']:>7s} mean {r['mean']:>7s} "
              f"1-a {r['sat']:>5s}")


if __name__ == "__main__":
    main()
