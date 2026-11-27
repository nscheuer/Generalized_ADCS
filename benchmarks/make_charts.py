"""Turn ``results/results.json`` into README/docs graphics (light and dark SVG).

Run in the GenADCS 3.0 environment (it has matplotlib)::

    .bench/genadcs3/bin/python benchmarks/make_charts.py [results.json] [out_dir]

Writes, for each theme ``light``/``dark``:

* ``speed-<theme>.svg``    small multiples: steps/s per tool for each scenario,
                           plus final-position error after one orbit
* ``scaling-<theme>.svg``  wall-clock time vs simulated duration (log-log)
* ``speedup-<theme>.svg``  speed-up over Generalized ADCS 2.x, with the 3.0
                           backends grouped in a dashed box
and ``table.md`` (Markdown) / ``table.rst`` (reST) with every headline number.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent

ORDER = ["genadcs3-rust", "genadcs3-cpp", "genadcs3-python", "genadcs2", "basilisk", "orekit"]
LABEL = {
    "genadcs3-rust": "Rust",
    "genadcs3-cpp": "C++",
    "genadcs3-python": "Python",
    "genadcs2": "2.x",
    "basilisk": "Basilisk",
    "orekit": "Orekit",
}
LEGEND = {
    "genadcs3-rust": "Generalized ADCS 3.0 · Rust",
    "genadcs3-cpp": "Generalized ADCS 3.0 · C++",
    "genadcs3-python": "Generalized ADCS 3.0 · Python (Numba)",
    "genadcs2": "Generalized ADCS 2.x",
    "basilisk": "Basilisk",
    "orekit": "Orekit",
}
OURS = {"genadcs3-rust", "genadcs3-cpp", "genadcs3-python"}

THEMES = {
    "light": dict(bg="#ffffff", fg="#14202e", soft="#4f5d6e", faint="#8a96a6", grid="#e6eaf0",
                  ours=["#0052a3", "#0066cc", "#3d8ce0"], prev="#a9c8ec",
                  others={"basilisk": "#b5bcc6", "orekit": "#d3d8de"}, box="#0066cc"),
    "dark": dict(bg="#0f1620", fg="#e5ecf4", soft="#a3b1c2", faint="#6f7e91", grid="#253244",
                 ours=["#80c0ff", "#4d9eff", "#2f7bd6"], prev="#3a5578",
                 others={"basilisk": "#5d6876", "orekit": "#465160"}, box="#80c0ff"),
}
FONT = ["Inter", "Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]


def color(key, t):
    if key in OURS:
        return t["ours"][["genadcs3-rust", "genadcs3-cpp", "genadcs3-python"].index(key)]
    if key == "genadcs2":
        return t["prev"]
    return t["others"][key]


def setup(t):
    plt.rcParams.update({
        "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": FONT,
        "font.size": 11, "axes.edgecolor": t["grid"], "axes.labelcolor": t["soft"],
        "xtick.color": t["soft"], "ytick.color": t["faint"], "text.color": t["fg"],
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
    })


def gradient_bar(ax, x, h, w, c, t, log=False, bottom=None):
    """Bar with a vertical gradient from ``c`` (top) to a faded tone (base)."""
    bar = ax.bar(x, h, w, color="none", edgecolor="none", bottom=bottom)[0]
    cmap = LinearSegmentedColormap.from_list("g", [matplotlib.colors.to_rgba(c, 0.35),
                                                   matplotlib.colors.to_rgba(c, 1.0)])
    y0 = bottom if bottom is not None else (ax.get_ylim()[0] if log else 0.0)
    img = ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), cmap=cmap, origin="lower",
                    extent=(x - w / 2, x + w / 2, y0, h), aspect="auto", zorder=2)
    img.set_clip_path(bar)
    return bar


def headline(runs, scenario):
    last = max(r["fraction"] for r in runs if r.get("final"))
    return {r["key"]: r for r in runs
            if r.get("final") and r["scenario"] == scenario and r["fraction"] == last}


def fmt_rate(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k" if v >= 1e4 else f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def fmt_err(v):
    if v == 0:
        return "0"
    if v >= 1:
        return f"{v:.0f} m" if v >= 10 else f"{v:.1f} m"
    exp = math.floor(math.log10(v))
    return f"{v / 10 ** exp:.0f}e{exp} m"


def footer(fig, meta, t, extra=""):
    m = meta
    host = f"{m['cpu']} · {m['logical_cpus']} threads · {m['os']}" + (" (WSL2)" if m.get("wsl") else "")
    sc = m["scenario"]
    text = (f"500 km, {sc['inclination_deg']}° circular orbit, J2 gravity, torque-free 12 kg spacecraft · "
            f"fixed-step RK4, dt = {sc['dt_s']} s, state recorded every step · median of ≥{m['repeats']} runs · "
            f"{host} · {m['date_utc'][:10]} · commit {m.get('git_commit')}")
    import textwrap
    width = int(fig.get_figwidth() * 16)
    lines = textwrap.wrap(text, width) + (textwrap.wrap(extra, width) if extra else [])
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, color=t["faint"], ha="left", va="bottom",
             linespacing=1.4)


def bracket(ax, i0, i1, t):
    """Bracket below the x tick labels grouping the Generalized ADCS 3.0 bars."""
    trans = ax.get_xaxis_transform()
    y = -0.13
    ax.plot([i0 - 0.35, i0 - 0.35, i1 + 0.35, i1 + 0.35], [y + 0.025, y, y, y + 0.025],
            transform=trans, color=t["box"], lw=1.2, clip_on=False)
    ax.text((i0 + i1) / 2, y - 0.02, "Generalized ADCS 3.0", transform=trans, ha="center", va="top",
            fontsize=9, fontweight="bold", color=t["box"], clip_on=False)


def bar_panel(ax, data, value, t, title, subtitle, log=True, fmt=fmt_rate, lower_better=False,
              unsupported=()):
    keys = [k for k in ORDER if k in data or k in unsupported]
    xs = np.arange(len(keys))
    vals = [value(data[k]) if k in data else None for k in keys]
    present = [v for v in vals if v]
    if log:
        lo, hi = min(present), max(present)
        ax.set_yscale("log")
        ax.set_ylim(lo / 4, hi * 6)
    else:
        ax.set_ylim(0, max(present) * 1.25)
    best = (min if lower_better else max)(present)
    for x, k, v in zip(xs, keys, vals):
        if v is None:
            ax.text(x, ax.get_ylim()[0] * (1.6 if log else 1), "not\nsupported", ha="center",
                    va="bottom", fontsize=8.5, color=t["faint"], style="italic")
            continue
        c = color(k, t)
        if k in OURS:
            gradient_bar(ax, x, v, 0.72, c, t, log=log)
        else:
            ax.bar(x, v, 0.72, color=c, zorder=2)
        bold = k in OURS and v == best
        ax.text(x, v * (1.25 if log else 1.02), fmt(v), ha="center", va="bottom", zorder=3,
                fontsize=10.5 if bold else 9.5, fontweight="bold" if (k in OURS) else "normal",
                color=c if k in OURS else t["soft"])
    ax.set_xticks(xs, [LABEL[k] for k in keys], fontsize=9.5)
    for lbl, k in zip(ax.get_xticklabels(), keys):
        if k in OURS:
            lbl.set_fontweight("bold")
            lbl.set_color(t["fg"])
    ours_idx = [i for i, k in enumerate(keys) if k in OURS]
    if ours_idx:
        bracket(ax, ours_idx[0], ours_idx[-1], t)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["fg"], pad=22)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=t["soft"])


def speed_chart(res, t, out):
    runs, meta = res["runs"], res["meta"]
    coupled, orbit = headline(runs, "coupled"), headline(runs, "orbit")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    bar_panel(axes[0], coupled, lambda r: r["steps_per_s"], t, "Attitude + orbit",
              "Simulation steps per second · higher is better", unsupported=("orekit",))
    bar_panel(axes[1], orbit, lambda r: r["steps_per_s"], t, "Orbit only",
              "Simulation steps per second · higher is better")
    acc = {k: r for k, r in coupled.items() if r.get("errors")}
    bar_panel(axes[2], acc, lambda r: max(r["errors"]["position_m"], 1e-12), t, "Accuracy after one orbit",
              "Final position error vs. high-precision reference · lower is better",
              fmt=fmt_err, lower_better=True, unsupported=("orekit",))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.84, bottom=0.27, wspace=0.2)
    footer(fig, meta, t, "GenADCS 3.0 and Basilisk always integrate attitude with the orbit; Orekit has no "
                         "rigid-body attitude dynamics; 2.x integrates attitude with SciPy RK45 (rtol 1e-7).")
    fig.savefig(out)
    plt.close(fig)


def scaling_chart(res, t, out):
    runs, meta = res["runs"], res["meta"]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    period = meta["scenario"]["orbit_period_s"]
    best_key, best_y = None, None
    for k in ORDER:
        pts = sorted((r["fraction"], r["median_s"]) for r in runs
                     if r["key"] == k and r["scenario"] == "coupled" and r.get("final"))
        if not pts:
            continue
        x = [p[0] * period / 60 for p in pts]
        y = [p[1] for p in pts]
        c = color(k, t)
        ours = k in OURS
        ax.plot(x, y, "-o", color=c, lw=3 if ours else 1.8, ms=7 if ours else 5,
                markeredgecolor=t["bg"], markeredgewidth=1.5, zorder=3 if ours else 2,
                label=f"{LEGEND[k]}  ·  {y[-1]:.3g} s per orbit")
        if best_y is None or y[-1] < best_y:
            best_key, best_y = k, y[-1]
    if "genadcs2" in [r["key"] for r in runs] and best_key:
        slow = [r for r in runs if r["key"] == "genadcs2" and r["scenario"] == "coupled"
                and r.get("final") and r["fraction"] == max(meta["scenario"]["fractions"])]
        if slow:
            factor = slow[0]["median_s"] / best_y
            ax.annotate(f"{LEGEND[best_key]}\n{factor:,.0f}× faster than 2.x",
                        xy=(period / 60, best_y), xytext=(-230, 18), textcoords="offset points",
                        fontsize=10.5, fontweight="bold", color=color(best_key, t),
                        arrowprops=dict(arrowstyle="->", color=color(best_key, t), lw=1.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    fracs = meta["scenario"]["fractions"]
    ax.set_xticks([f * period / 60 for f in fracs],
                  [f"{f * period / 60:.0f} min" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("Simulated time (1/8, 1/4, 1/2 and 1 orbit)")
    ax.set_ylabel("Wall-clock time [s]")
    ax.grid(color=t["grid"], linestyle="--", linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Run time vs. simulated duration · attitude + orbit", loc="left", fontsize=14,
                 fontweight="bold", color=t["fg"], pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    labelcolor=t["soft"])
    for txt in leg.get_texts():
        if txt.get_text().startswith("Generalized ADCS 3.0"):
            txt.set_fontweight("bold")
            txt.set_color(t["fg"])
    fig.subplots_adjust(left=0.07, right=0.64, top=0.9, bottom=0.22)
    footer(fig, meta, t, "Lower is better. Each point is one run of the given length after warm-up.")
    fig.savefig(out)
    plt.close(fig)


def speedup_chart(res, t, out):
    runs, meta = res["runs"], res["meta"]
    groups = [("coupled", "Attitude + orbit"), ("orbit", "Orbit only")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (scenario, title) in zip(axes, groups):
        data = headline(runs, scenario)
        base = data.get("genadcs2")
        if not base:
            continue
        keys = [k for k in ORDER if k in data]
        xs = np.arange(len(keys))
        vals = [base["median_s"] / data[k]["median_s"] for k in keys]
        ax.set_yscale("log")
        ax.set_ylim(min(vals) / 3, max(vals) * 20)
        for x, k, v in zip(xs, keys, vals):
            c = color(k, t)
            if k in OURS:
                gradient_bar(ax, x, v, 0.6, c, t, log=True)
            else:
                ax.bar(x, v, 0.6, color=c, zorder=2)
            ax.text(x, v * 1.2, f"{v:,.0f}×" if v >= 10 else f"{v:.1f}×", ha="center", va="bottom",
                    fontsize=10, fontweight="bold" if k in OURS else "normal",
                    color=c if k in OURS else t["soft"])
        ours_idx = [i for i, k in enumerate(keys) if k in OURS]
        if ours_idx:
            lo, hi = ours_idx[0] - 0.45, ours_idx[-1] + 0.45
            y0 = ax.get_ylim()[0]
            top = max(vals[i] for i in ours_idx) * 12
            ax.add_patch(FancyBboxPatch((lo, y0 * 1.05), hi - lo, top - y0 * 1.05, boxstyle="square,pad=0",
                                        fill=True, facecolor=matplotlib.colors.to_rgba(t["box"], 0.05),
                                        edgecolor=t["box"], linestyle="--", linewidth=1.3, zorder=1))
            ax.text((lo + hi) / 2, top / 1.25, "Generalized ADCS 3.0", ha="center", va="top",
                    fontsize=10, fontweight="bold", color=t["box"])
        ax.axhline(1.0, color=t["faint"], lw=1, ls=":", zorder=1)
        ax.set_xticks(xs, [LABEL[k] for k in keys], fontsize=9.5)
        for lbl, k in zip(ax.get_xticklabels(), keys):
            if k in OURS:
                lbl.set_fontweight("bold")
                lbl.set_color(t["fg"])
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["fg"], pad=10)
    axes[0].set_ylabel("Speed-up over Generalized ADCS 2.x")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2, wspace=0.08)
    footer(fig, meta, t, "Higher is better. 1× (dotted line) = Generalized ADCS 2.x.")
    fig.savefig(out)
    plt.close(fig)


def tables(res, outdir):
    runs, meta = res["runs"], res["meta"]
    rows = []
    for k in ORDER:
        row = [LEGEND[k]]
        for scenario in ("coupled", "orbit"):
            r = headline(runs, scenario).get(k)
            row.append(f"{r['steps_per_s']:,.0f}" if r else "—")
        r = headline(runs, "coupled").get(k) or headline(runs, "orbit").get(k)
        e = (r or {}).get("errors", {})
        row.append(f"{e['position_m']:.2e}" if "position_m" in e else "—")
        row.append(f"{e['attitude_deg']:.1e}" if "attitude_deg" in e else "—")
        rows.append(row)
    head = ["Tool", "Attitude + orbit [steps/s]", "Orbit only [steps/s]",
            "Position error, 1 orbit [m]", "Attitude error, 1 orbit [deg]"]
    md = ["| " + " | ".join(head) + " |", "|" + "|".join([":--"] + [":-:"] * 4) + "|"]
    for row in rows:
        cells = [f"**{c}**" if row[0].startswith("Generalized ADCS 3.0") else c for c in row]
        md.append("| " + " | ".join(cells) + " |")
    (outdir / "table.md").write_text("\n".join(md) + "\n")
    rst = [".. list-table::", "   :header-rows: 1", "   :widths: 34 18 18 15 15", ""]
    for i, row in enumerate([head] + rows):
        for j, c in enumerate(row):
            rst.append(("   * - " if j == 0 else "     - ") + (f"**{c}**" if i and row[0].startswith(
                "Generalized ADCS 3.0") and j == 0 else c))
    (outdir / "table.rst").write_text("\n".join(rst) + "\n")


def readme_block(res, outdir):
    """Summary sentence + table for the README, generated from the data."""
    runs, meta = res["runs"], res["meta"]
    coupled = headline(runs, "coupled")
    ours = {k: v for k, v in coupled.items() if k in OURS}
    lines = []
    if ours and "genadcs2" in coupled:
        best = min(ours, key=lambda k: ours[k]["median_s"])
        factor = coupled["genadcs2"]["median_s"] / ours[best]["median_s"]
        period_min = meta["scenario"]["orbit_period_s"] / 60
        lines.append(
            f"> One full orbit ({period_min:.0f} simulated minutes, {ours[best]['steps']:,} steps) with attitude "
            f"takes **{ours[best]['median_s']:.3g} s** on the {LEGEND[best].split('· ')[-1]} backend, "
            f"**{factor:,.0f}× faster** than Generalized ADCS 2.x "
            f"({meta['cpu']}, {meta['date_utc'][:10]}).")
        lines.append("")
    lines.append((outdir / "table.md").read_text().rstrip())
    return "\n".join(lines)


def update_readme(res, outdir):
    readme = HERE.parent / "README.md"
    start, end = "<!-- benchmarks:start -->", "<!-- benchmarks:end -->"
    text = readme.read_text()
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    note = "<!-- Generated by benchmarks/make_charts.py from benchmarks/results/results.json; do not edit by hand. -->"
    readme.write_text(f"{head}{start}\n{note}\n\n{readme_block(res, outdir)}\n\n{end}{tail}")
    print("updated the benchmark block in README.md")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "results.json"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    res = json.loads(src.read_text())
    for name, t in THEMES.items():
        setup(t)
        speed_chart(res, t, outdir / f"speed-{name}.svg")
        scaling_chart(res, t, outdir / f"scaling-{name}.svg")
        speedup_chart(res, t, outdir / f"speedup-{name}.svg")
    tables(res, outdir)
    print("wrote", ", ".join(sorted(p.name for p in outdir.iterdir() if p.suffix in (".svg", ".md", ".rst"))))
    if len(sys.argv) <= 2:  # default location: also publish into the docs and README
        update_readme(res, outdir)
        docs = HERE.parent / "docs" / "source" / "_static" / "benchmarks"
        docs.mkdir(parents=True, exist_ok=True)
        for p in outdir.iterdir():
            if p.suffix == ".svg":
                (docs / p.name).write_bytes(p.read_bytes())
            elif p.suffix == ".rst":  # .inc so Sphinx does not treat it as a page
                (docs / "table.inc").write_bytes(p.read_bytes())
        print("copied charts and table.rst to", docs.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
