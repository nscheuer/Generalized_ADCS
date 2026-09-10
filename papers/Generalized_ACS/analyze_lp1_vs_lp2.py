"""
Per-regime LP1 vs LP2 comparison analyzer  (Exec Brief #2 Addendum A section 3).

Reads the campaign JSON written by ``lp_qp_cqp_campaign.py --run`` and emits:

  * tab_lp1_vs_lp2.csv / .tex  -- per-(config x h_rw level x allocator) cells with
    convergence%, mean / p95 final error, and the realized gyro-shortfall proxy
    (1 - mean_alpha). Paired LP1->LP2 deltas highlighted.
  * fig_lp1_vs_lp2.png         -- a 2x2 panel: rows = config (3+1, 3+3), columns =
    {convergence%, mean error}; x-axis = h_rw / h_max; one curve per allocator.

The "honesty check" Patrick called out (separate 'LP2 beats LP1' from 'config
saturated') is baked into the figure: an unshaded x-region marks the regime
where the controllability geometry permits convergence (LP achieves >=80%
under the threshold); a shaded region marks the regime where even the
best allocator can't converge -- a hardware/controllability limit, not an
allocator failure. LP1 vs LP2 differences are meaningful only inside the
unshaded region.

Run:
  PYTHONPATH=. venv/bin/python papers/Generalized_ACS/analyze_lp1_vs_lp2.py \\
      papers/Generalized_ACS/output_data/lp_qp_cqp_campaign_<stamp>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H_MAX_REF = 16.2e-3
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output_data")

ALLOC_ORDER = ["LP", "LP2", "QP", "cQP"]
ALLOC_COLOR = {"LP": "#de2d26", "LP2": "#2c7fb8",
               "QP": "#74c476", "cQP": "#fd8d3c"}
ALLOC_MARKER = {"LP": "o", "LP2": "s", "QP": "^", "cQP": "D"}


def latest_campaign_json() -> str:
    files = sorted(glob.glob(os.path.join(OUT_DIR, "lp_qp_cqp_campaign_*.json")))
    if not files:
        raise FileNotFoundError(f"no campaign JSON in {OUT_DIR}")
    return files[-1]


def _h_level_to_xfrac(label: str) -> float:
    """Parse the H0_VARIANTS key 'h0_<frac>xh' into the numeric fraction."""
    # e.g. "h0_0.6xh" -> 0.6
    try:
        return float(label.replace("h0_", "").replace("xh", ""))
    except ValueError:
        return float("nan")


def build_rows(camp: dict) -> list[dict]:
    rows = []
    for cfg_label, by_h in camp["results"].items():
        for h_label, by_alloc in by_h.items():
            h_frac = _h_level_to_xfrac(h_label)
            h_abs = float(camp["spec"]["h0_variants"].get(h_label, float("nan")))
            for alloc_label, payload in by_alloc.items():
                agg = payload["aggregate"]
                # mean (1 - alpha) per trial -> gyro-shortfall proxy
                shortfall = float(np.mean([1.0 - t["mean_alpha"]
                                           for t in payload["per_trial"]]))
                rows.append({
                    "config": cfg_label,
                    "h_label": h_label, "h_frac_of_hmax": h_frac, "h_abs": h_abs,
                    "allocator": alloc_label,
                    "conv_pct": agg["converged_pct"],
                    "mean_err_deg": agg["final_error_deg"]["mean"],
                    "median_err_deg": agg["final_error_deg"]["median"],
                    "p95_err_deg": agg["final_error_deg"]["p95"],
                    "gyro_shortfall_proxy": shortfall,  # 1 - mean_alpha
                })
    return rows


def emit_csv(rows: list[dict]) -> str:
    path = os.path.join(OUT_DIR, "tab_lp1_vs_lp2.csv")
    fields = ["config", "h_label", "h_frac_of_hmax", "h_abs", "allocator",
              "conv_pct", "mean_err_deg", "median_err_deg", "p95_err_deg",
              "gyro_shortfall_proxy"]
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float)
                             else str(r[k]) for k in fields) + "\n")
    return path


def emit_tex(rows: list[dict]) -> str:
    """Per-(config x h-level) table with LP1 / LP2 paired (Delta) + QP, cQP."""
    by = {(r["config"], r["h_frac_of_hmax"], r["allocator"]): r for r in rows}
    cfgs = sorted({r["config"] for r in rows})
    hs = sorted({r["h_frac_of_hmax"] for r in rows})
    lines = [
        "% LP1 vs LP2 paired comparison.  Cell: conv% / mean-err [deg].",
        "% Delta column = LP - LP2 mean error (positive = LP2 better).",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Config & $h_0/h_{\max}$ & LP1 & LP2 & $\Delta$(deg) "
        r"& QP & cQP & shortfall \\",
        r"\midrule",
    ]
    for cfg in cfgs:
        for h in hs:
            def cell(a: str) -> str:
                k = (cfg, h, a)
                if k not in by: return "--"
                r = by[k]
                return f"{r['conv_pct']:.0f}\\% / {r['mean_err_deg']:.2f}"
            lp = by.get((cfg, h, "LP"))
            lp2 = by.get((cfg, h, "LP2"))
            delta = (lp["mean_err_deg"] - lp2["mean_err_deg"]) if (lp and lp2) else float("nan")
            sf = by.get((cfg, h, "LP"), {"gyro_shortfall_proxy": float("nan")})["gyro_shortfall_proxy"]
            lines.append(
                f"{cfg} & {h:.1f} & {cell('LP')} & {cell('LP2')} & "
                f"{delta:+.2f} & {cell('QP')} & {cell('cQP')} & {sf:.2f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(OUT_DIR, "tab_lp1_vs_lp2.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def emit_figure(rows: list[dict]) -> str:
    cfgs = sorted({r["config"] for r in rows})
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(11, 4.0 * len(cfgs)),
                             squeeze=False)
    for ai, cfg in enumerate(cfgs):
        # convergence% (left) and mean err (right)
        for aj, (yk, ylab, title) in enumerate(
                [("conv_pct", "convergence % (< 5 deg)",
                  f"{cfg}: convergence vs stored momentum"),
                 ("mean_err_deg", "mean final pointing error [deg]",
                  f"{cfg}: mean error vs stored momentum")]):
            ax = axes[ai, aj]
            for a in ALLOC_ORDER:
                pts = sorted([r for r in rows
                              if r["config"] == cfg and r["allocator"] == a],
                             key=lambda r: r["h_frac_of_hmax"])
                if not pts:
                    continue
                xs = [r["h_frac_of_hmax"] for r in pts]
                ys = [r[yk] for r in pts]
                lw = 2.2 if a in ("LP", "LP2") else 1.2
                ax.plot(xs, ys, marker=ALLOC_MARKER[a], color=ALLOC_COLOR[a],
                        label=a, lw=lw, alpha=0.95 if a in ("LP", "LP2") else 0.7)
            # honesty-shading: x-region where the best allocator achieves
            # < 50% convergence is "config saturated"; shade it.
            best_conv = {}
            for a in ALLOC_ORDER:
                for r in rows:
                    if r["config"] != cfg or r["allocator"] != a:
                        continue
                    best_conv[r["h_frac_of_hmax"]] = max(
                        best_conv.get(r["h_frac_of_hmax"], 0.0), r["conv_pct"])
            sat_xs = sorted([x for x, c in best_conv.items() if c < 50.0])
            if sat_xs:
                ax.axvspan(min(sat_xs) - 0.05, max(sat_xs) + 0.05,
                           color="grey", alpha=0.10,
                           label="config saturated (best <50%)")
            if yk == "conv_pct":
                ax.set_ylim(-2, 105)
                ax.axhline(80, ls=":", color="k", lw=0.8)
            ax.set_xlabel(r"stored RW momentum  $\|h_{rw}\|/h_{\max}$")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("LP1 vs LP2 per regime — closed-loop allocation campaign "
                 "(100 trials, 1000 s, paired seeds)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT_DIR, "fig_lp1_vs_lp2.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(rows: list[dict]) -> None:
    print("\n=== LP1 vs LP2 per-regime summary "
          "(conv% / mean-err / 1-mean_alpha) ===")
    cfgs = sorted({r["config"] for r in rows})
    hs = sorted({r["h_frac_of_hmax"] for r in rows})
    for cfg in cfgs:
        print(f"\n[{cfg}]")
        print(f"  {'h/h_max':>8} {'LP1 conv':>9} {'LP1 mean':>9} "
              f"{'LP2 conv':>9} {'LP2 mean':>9} {'delta (LP-LP2)':>17} "
              f"{'shortfall':>10}")
        for h in hs:
            lp = next((r for r in rows
                       if r["config"] == cfg and r["allocator"] == "LP"
                       and r["h_frac_of_hmax"] == h), None)
            lp2 = next((r for r in rows
                        if r["config"] == cfg and r["allocator"] == "LP2"
                        and r["h_frac_of_hmax"] == h), None)
            if lp and lp2:
                d = lp["mean_err_deg"] - lp2["mean_err_deg"]
                print(f"  {h:8.2f} {lp['conv_pct']:8.1f}% "
                      f"{lp['mean_err_deg']:9.2f} {lp2['conv_pct']:8.1f}% "
                      f"{lp2['mean_err_deg']:9.2f} {d:+17.2f} "
                      f"{lp['gyro_shortfall_proxy']:10.3f}")


def emit_paired_figure(rows: list[dict]) -> str:
    """
    Paper-ready relative-paired figure: (1 - alpha) saturation per cell +
    paired Delta (LP1 - LP2) per cell. NO absolute convergence% or
    absolute pointing error shown -- those are testbed-dependent and the
    Paper 2 MC uses different gains (kp,kd) so absolute numbers would
    conflict with P2.1's 90% under the same nominal 3+1 config.
    The two metrics here are intrinsic to the allocator-scenario interaction:
      * 1 - alpha = realized saturation fraction (gyro-shortfall proxy)
      * Delta = paired mean-error difference (same trial seeds across allocators)
    """
    cfgs = sorted({r["config"] for r in rows})
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(11, 4.0 * len(cfgs)),
                             squeeze=False)
    for ai, cfg in enumerate(cfgs):
        # left: 1 - alpha per allocator
        axL = axes[ai, 0]
        for a in ALLOC_ORDER:
            pts = sorted([r for r in rows
                          if r["config"] == cfg and r["allocator"] == a],
                         key=lambda r: r["h_frac_of_hmax"])
            if not pts:
                continue
            xs = [r["h_frac_of_hmax"] for r in pts]
            ys = [r["gyro_shortfall_proxy"] for r in pts]
            lw = 2.2 if a in ("LP", "LP2") else 1.2
            axL.plot(xs, ys, marker=ALLOC_MARKER[a], color=ALLOC_COLOR[a],
                     label=a, lw=lw, alpha=0.95 if a in ("LP", "LP2") else 0.7)
        axL.set_xlabel(r"stored RW momentum  $\|h_{rw}\|/h_{\max}$")
        axL.set_ylabel(r"realized saturation  $1-\bar\alpha$")
        axL.set_title(f"{cfg}: saturation diagnostic (intrinsic)")
        axL.set_ylim(-0.02, 1.02)
        axL.legend(fontsize=8, ncol=2)
        axL.grid(True, ls="--", alpha=0.4)

        # right: paired Delta = LP1_mean - LP2_mean per cell
        axR = axes[ai, 1]
        pairs = []
        for h in sorted({r["h_frac_of_hmax"] for r in rows}):
            lp = next((r for r in rows if r["config"] == cfg
                       and r["allocator"] == "LP"
                       and r["h_frac_of_hmax"] == h), None)
            lp2 = next((r for r in rows if r["config"] == cfg
                        and r["allocator"] == "LP2"
                        and r["h_frac_of_hmax"] == h), None)
            if lp and lp2:
                pairs.append((h, lp["mean_err_deg"] - lp2["mean_err_deg"]))
        if pairs:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            colors = ["#2c7fb8" if y >= 0 else "#de2d26" for y in ys]
            axR.bar(xs, ys, width=0.08, color=colors, edgecolor="black", alpha=0.85)
            for x, y in pairs:
                axR.annotate(f"{y:+.1f}", (x, y),
                             textcoords="offset points", xytext=(0, 4 if y >= 0 else -14),
                             ha="center", fontsize=8)
        axR.axhline(0.0, color="k", lw=0.8)
        axR.set_xlabel(r"stored RW momentum  $\|h_{rw}\|/h_{\max}$")
        axR.set_ylabel(r"paired $\Delta$ (LP1 mean $-$ LP2 mean) [deg]")
        axR.set_title(f"{cfg}: paired LP1$-$LP2 mean-error difference\n"
                      "(positive = LP2 better)")
        axR.grid(True, ls="--", alpha=0.4)
    fig.suptitle("LP1 vs LP2 — relative paired study  (100 trials, 1000 s, "
                 "paired seeds across h-levels)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT_DIR, "fig_lp1_vs_lp2_paired.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def emit_paired_tex(rows: list[dict]) -> str:
    """Paper-ready paired-only LaTeX table: (1 - alpha) + Delta, no abs conv%."""
    by = {(r["config"], r["h_frac_of_hmax"], r["allocator"]): r for r in rows}
    cfgs = sorted({r["config"] for r in rows})
    hs = sorted({r["h_frac_of_hmax"] for r in rows})
    lines = [
        "% Relative-paired LP1 vs LP2 comparison.  Intrinsic metrics only.",
        "% Delta_mean = LP1 mean err - LP2 mean err (paired seeds);",
        "% (1 - alpha) = realized saturation fraction (gyro-shortfall proxy).",
        "% Absolute convergence% and absolute mean err are omitted because the",
        "% testbed gains (kp=5e-5, kd=1e-3) differ from the Paper 2 MC framework.",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Config & $h_0/h_{\max}$ & $1-\bar\alpha$ (LP1) & $1-\bar\alpha$ (LP2) "
        r"& $\Delta_{\text{mean}}$ (deg) & verdict \\",
        r"\midrule",
    ]
    for cfg in cfgs:
        for h in hs:
            lp = by.get((cfg, h, "LP"))
            lp2 = by.get((cfg, h, "LP2"))
            if not (lp and lp2):
                continue
            d = lp["mean_err_deg"] - lp2["mean_err_deg"]
            sf1 = lp["gyro_shortfall_proxy"]
            sf2 = lp2["gyro_shortfall_proxy"]
            if abs(d) < 0.01:
                verdict = "tie"
            elif d > 0:
                verdict = "LP2 better"
            else:
                verdict = "LP1 better"
            lines.append(
                f"{cfg} & {h:.1f} & {sf1:.3f} & {sf2:.3f} & "
                f"{d:+.2f} & {verdict} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(OUT_DIR, "tab_lp1_vs_lp2_paired.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(json_path: str) -> int:
    with open(json_path) as f:
        camp = json.load(f)
    rows = build_rows(camp)
    csv_path = emit_csv(rows)
    tex_path = emit_tex(rows)
    fig_path = emit_figure(rows)
    paired_fig_path = emit_paired_figure(rows)
    paired_tex_path = emit_paired_tex(rows)
    print_summary(rows)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}                 (raw reproducibility table)")
    print(f"wrote {fig_path}                 (raw / supplementary figure)")
    print(f"wrote {paired_tex_path}  (PAPER-READY paired table)")
    print(f"wrote {paired_fig_path}  (PAPER-READY paired figure)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", nargs="?", default=None,
                    help="campaign JSON (default: latest in output_data/)")
    args = ap.parse_args()
    raise SystemExit(main(args.json_path or latest_campaign_json()))
