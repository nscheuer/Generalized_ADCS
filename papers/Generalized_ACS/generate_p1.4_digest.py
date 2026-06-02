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

    # --- P1.1 LP1 vs LP2 (allocation), paired & clustered per config/h ---
    # 'LP' in the source is the single-stage allocator (LP1); 'LP2' is the
    # two-stage allocator. Emit them as adjacent LP1/LP2 rows at representative
    # h levels so the paired LP1-vs-LP2 delta and 1-abar are read off directly.
    lp = read_csv("tab_lp1_vs_lp2.csv")
    by = {(r["config"], r["h_frac_of_hmax"], r["allocator"]): r for r in lp}
    for cfg in ("3+1", "3+3"):
        for h in ("0.0000", "0.9000"):
            r1, r2 = by.get((cfg, h, "LP")), by.get((cfg, h, "LP2"))
            if r1 is None or r2 is None:
                continue
            d = float(r1["mean_err_deg"]) - float(r2["mean_err_deg"])  # LP1 - LP2
            hlab = "h0" if h == "0.0000" else "h.9"
            rows.append({
                "exp": "P1.1 alloc", "config": f"{cfg} ({hlab})",
                "controller": "LP1", "alloc": "1-stage LP", "task": "vector",
                "n": "100", "conv": f(r1["conv_pct"]) + "%",
                "mean": f(r1["mean_err_deg"]), "p95": f(r1["p95_err_deg"]),
                "sat": f(r1["gyro_shortfall_proxy"]),
                "basis": "absolute (harsh testbed); paired w/ LP2 below",
            })
            rows.append({
                "exp": "P1.1 alloc", "config": f"{cfg} ({hlab})",
                "controller": "LP2", "alloc": "2-stage LP", "task": "vector",
                "n": "100", "conv": f(r2["conv_pct"]) + "%",
                "mean": f(r2["mean_err_deg"]), "p95": f(r2["p95_err_deg"]),
                "sat": f(r2["gyro_shortfall_proxy"]),
                "basis": f"paired Δ(LP1-LP2) mean {d:+.2f} deg",
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
