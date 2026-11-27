"""Draw the README graphics (light and dark SVG) from real data.

Run from the repository root in the GenADCS 3.0 environment::

    .bench/genadcs3/bin/python assets/readme_figures.py

* ``hero-<theme>.svg``         headline numbers, read from benchmarks/results/results.json
* ``timeline-<theme>.svg``     pointing error and body rate from the bundled
                               mode-switching example (B-dot -> Sun -> nadir), run here
* ``architecture-<theme>.svg`` how a GenADCS 3.0 simulation is put together

Every number shown is computed in this script from those two sources.
"""

from __future__ import annotations

import json
import math
import runpy
import warnings
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
RESULTS = ROOT / "benchmarks" / "results" / "results.json"
EXAMPLE = ROOT / "GenADCS/examples/hardware/ADCS_subsystem/controllers/example_mode_switching_attitude.py"

THEMES = {
    "light": dict(bg="#fbfcfe", card="#f3f6fb", fg="#14202e", soft="#4f5d6e", faint="#8a96a6",
                  line="#dfe5ee", accent="#0066cc", accent2="#3d8ce0", accent3="#8ab8ec",
                  accentSoft="#e8f1fc", grey="#b5bcc6", prev="#a9c8ec",
                  b1="#e8f1fc", b2="#fdf1e3", b3="#e7f6ed", warm="#a35a00", ok="#1a7f4b"),
    "dark": dict(bg="#0f1620", card="#162130", fg="#e5ecf4", soft="#a3b1c2", faint="#6f7e91",
                 line="#253244", accent="#80c0ff", accent2="#4d9eff", accent3="#2f7bd6",
                 accentSoft="#15263b", grey="#5d6876", prev="#3a5578",
                 b1="#15263b", b2="#2b2114", b3="#13291e", warm="#f0b46c", ok="#5cc98d"),
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Inter, Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, Consolas, 'Liberation Mono', monospace"


# --------------------------------------------------------------------------- data
def benchmark_numbers():
    res = json.loads(RESULTS.read_text())
    last = max(r["fraction"] for r in res["runs"] if r.get("final"))
    head = {(r["scenario"], r["key"]): r for r in res["runs"]
            if r.get("final") and r["fraction"] == last}
    rust = head[("coupled", "genadcs3-rust")]
    old = head[("coupled", "genadcs2")]
    ours = [head[("coupled", k)] for k in ("genadcs3-rust", "genadcs3-cpp", "genadcs3-python")]
    bars = [(label, head[("coupled", key)]["steps_per_s"], key)
            for label, key in (("3.0 · Rust", "genadcs3-rust"), ("3.0 · C++", "genadcs3-cpp"),
                               ("3.0 · Python", "genadcs3-python"), ("Basilisk", "basilisk"),
                               ("2.x", "genadcs2")) if ("coupled", key) in head]
    return {
        "speedup": old["median_s"] / rust["median_s"],
        "orbit_s": rust["median_s"],
        "steps": rust["steps"],
        "max_err_m": max(r["errors"]["position_m"] for r in ours),
        "bars": bars,
        "meta": res["meta"],
    }


def mission_timeline():
    warnings.filterwarnings("ignore")
    ns = runpy.run_path(str(EXAMPLE), run_name="readme_figures")
    result = ns["main"](show=False)
    sat = result.satellites["sat_000"]
    t = np.asarray(sat.times_s)
    q = np.asarray(sat.satellite_attitude)
    g = np.asarray(sat.goal_attitude)
    w = np.asarray(sat.satellite_rate)
    dot = np.abs(np.sum(q * g, axis=1))
    err = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))  # NaN while there is no goal
    rate = np.degrees(np.linalg.norm(w, axis=1))
    return t, err, rate


# --------------------------------------------------------------------------- hero
def fmt_err(m):
    if m < 1e-3:
        return f"{m * 1e6:.1f} µm"
    if m < 1:
        return f"{m * 1e3:.1f} mm"
    return f"{m:.1f} m"


def hero(t, n):
    W, H = 1280, 580
    tiles = [
        (f"{n['speedup']:,.0f}×", "faster than ADCS 2.x", "Rust backend vs. Generalized ADCS 2.x"),
        (f"{n['orbit_s']:.3f} s", "to simulate one full orbit", f"{n['steps']:,} RK4 steps · Rust backend"),
        (fmt_err(n["max_err_m"]), "position error after one orbit", "all three backends vs. DOP853 reference"),
    ]
    tx, ty, tw, th, gap = 56, 236, 360, 150, 24
    tiles_svg = "\n".join(f"""
  <rect x="{tx + i * (tw + gap)}" y="{ty}" width="{tw}" height="{th}" rx="14" fill="{t['card']}" stroke="{t['line']}"/>
  <text x="{tx + i * (tw + gap) + 24}" y="{ty + 66}" font-family="{SANS}" font-size="50" font-weight="800" letter-spacing="-1.5" fill="{t['accent']}">{escape(big)}</text>
  <text x="{tx + i * (tw + gap) + 24}" y="{ty + 100}" font-family="{SANS}" font-size="18" font-weight="600" fill="{t['fg']}">{escape(mid)}</text>
  <text x="{tx + i * (tw + gap) + 24}" y="{ty + 126}" font-family="{SANS}" font-size="14" fill="{t['soft']}">{escape(small)}</text>"""
                          for i, (big, mid, small) in enumerate(tiles))

    # Mini log-scale bar chart of steps per second.
    bx, by, bw = 56, 418, 1128
    lo, hi = math.log10(min(v for _, v, _ in n["bars"]) / 2), math.log10(max(v for _, v, _ in n["bars"]) * 1.4)
    rows = []
    for i, (label, v, key) in enumerate(n["bars"]):
        y = by + i * 22
        width = (math.log10(v) - lo) / (hi - lo) * (bw - 300)
        color = (t["accent"] if key == "genadcs3-rust" else t["accent2"] if key == "genadcs3-cpp"
                 else t["accent3"] if key == "genadcs3-python" else t["prev"] if key == "genadcs2" else t["grey"])
        weight = "700" if key.startswith("genadcs3") else "400"
        rows.append(f"""
  <text x="{bx}" y="{y + 13}" font-family="{SANS}" font-size="13" font-weight="{weight}" fill="{t['fg'] if weight == '700' else t['soft']}">{escape(label)}</text>
  <rect x="{bx + 110}" y="{y + 2}" width="{width:.1f}" height="14" rx="3" fill="{color}"/>
  <text x="{bx + 118 + width:.1f}" y="{y + 14}" font-family="{SANS}" font-size="12.5" font-weight="{weight}" fill="{t['soft']}">{v:,.0f} steps/s</text>""")
    meta = n["meta"]
    foot = (f"Torque-free 12 kg spacecraft, 500 km 51.6° orbit with J2, fixed-step RK4 at dt = 0.25 s · "
            f"{meta['cpu']} · {meta['date_utc'][:10]} · commit {meta.get('git_commit')}")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Generalized ADCS 3.0: {escape(tiles[0][0])} faster than 2.x, {escape(tiles[1][0])} per simulated orbit, {escape(tiles[2][0])} position error after one orbit.">
  <defs><pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.1" fill="{t['line']}"/></pattern></defs>
  <rect width="{W}" height="{H}" rx="18" fill="{t['bg']}"/>
  <rect width="{W}" height="{H}" rx="18" fill="url(#dots)" opacity="0.6"/>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t['line']}"/>
  <text x="56" y="84" font-family="{SANS}" font-size="15" font-weight="700" letter-spacing="2" fill="{t['accent']}">GENERALIZED ADCS 3.0</text>
  <text x="56" y="140" font-family="{SANS}" font-size="46" font-weight="800" letter-spacing="-1.5" fill="{t['fg']}">One spacecraft model. Three engines.</text>
  <text x="56" y="184" font-family="{SANS}" font-size="20" fill="{t['soft']}">The same simulation runs on Python (Numba), C++ and Rust backends; pick one with an import path.</text>
  {tiles_svg}
  <text x="{bx}" y="{by - 10}" font-family="{SANS}" font-size="13" font-weight="700" letter-spacing="1" fill="{t['faint']}">SIMULATION STEPS PER SECOND · ATTITUDE + ORBIT · LOG SCALE</text>
  {''.join(rows)}
  <text x="56" y="{H - 18}" font-family="{SANS}" font-size="12" fill="{t['faint']}">{escape(foot)}</text>
</svg>
"""


# --------------------------------------------------------------------------- timeline
def timeline(t, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time, err, rate = data
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "text.color": t["fg"], "axes.labelcolor": t["soft"], "xtick.color": t["soft"],
                         "ytick.color": t["faint"], "axes.edgecolor": t["line"],
                         "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"]})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.8, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})
    phases = [(0, 20, "B-dot detumble", t["b2"], t["warm"]), (20, 55, "Sun pointing", t["b1"], t["accent"]),
              (55, 90, "Nadir tracking", t["b3"], t["ok"])]
    for ax in (a1, a2):
        for x0, x1, _, fill, _ in phases:
            ax.axvspan(x0, x1, color=fill, zorder=0, lw=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=t["line"], lw=0.8, zorder=1)
    for x0, x1, name, _, c in phases:
        a1.text((x0 + x1) / 2, 1.02, name, transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=c)
    a1.semilogy(time, err, color=t["accent"], lw=2.6, zorder=3)
    a1.set_ylabel("Pointing error [deg]")
    for x0, x1, _, _, c in phases[1:]:
        # end-exclusive: at t = x1 the next mode's goal is already active
        m = (time >= x0) & (time < x1 if x1 < time[-1] else time <= x1) & np.isfinite(err)
        if m.any():
            a1.annotate(f"{err[m][0]:.0f}° → {err[m][-1]:.2f}°", (time[m][-1], err[m][-1]),
                        xytext=(-12, 26), textcoords="offset points", ha="right", fontsize=11,
                        fontweight="bold", color=c, arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    a1.text(10, 0.5, "no pointing goal\nwhile detumbling", ha="center", va="center", fontsize=10,
            color=t["faint"], transform=a1.get_xaxis_transform(), style="italic")
    a2.plot(time, rate, color=t["accent2"], lw=2.2, zorder=3)
    a2.set_ylabel("Body rate [deg/s]")
    a2.set_xlabel("Time [s]")
    a2.set_xlim(0, 90)
    fig.text(0.07, 0.95, "One satellite, three ADCS modes", fontsize=16, fontweight="bold", color=t["fg"])
    fig.text(0.07, 0.915, "ADCSStateMachine timeline from GenADCS/examples/.../example_mode_switching_attitude.py · "
                          "3 reaction wheels, 3 magnetorquers, IGRF-14", fontsize=10.5, color=t["soft"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.1)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- architecture
def architecture(t):
    W, H = 1280, 500

    def box(x, y, w, h, title, lines, fill=None, stroke=None, title_color=None):
        body = "\n".join(
            f'<text x="{x + 18}" y="{y + 58 + i * 21}" font-family="{SANS}" font-size="14" fill="{t["soft"]}">{line}</text>'
            for i, line in enumerate(lines))
        return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill or t['card']}" stroke="{stroke or t['line']}"/>
  <text x="{x + 18}" y="{y + 32}" font-family="{SANS}" font-size="18" font-weight="700" fill="{title_color or t['fg']}">{title}</text>
  {body}"""

    def arrow(x1, y1, x2, y2, color=None):
        c = color or t["faint"]
        a = math.atan2(y2 - y1, x2 - x1)
        p = lambda s: f"{x2 - 9 * math.cos(a + s):.1f},{y2 - 9 * math.sin(a + s):.1f}"
        return (f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{c}" stroke-width="1.8" fill="none"/>'
                f'<path d="M{x2} {y2} L{p(0.45)} L{p(-0.45)} Z" fill="{c}"/>')

    mono = lambda s: f'<tspan font-family="{MONO}" font-size="13" fill="{t["accent"]}">{s}</tspan>'
    left_x, lw = 32, 400
    inputs = [
        ("Satellite", [f"{mono('SatelliteGeometry')} mass, inertia, panels",
                       f"{mono('ADCSSubsystem')} wheels, magnetorquers,",
                       "sensors, estimator, controller, modes"], 106),
        ("Environment", ["Earth, J2–J6, IGRF-14 / WMM / CHAOS-8,", "NRLMSISE-00, DE421 ephemeris"], 86),
        ("DynamicsConfig", ["forces and torques: zonal gravity, drag,", "SRP, lunisolar, gravity gradient, dipole"], 86),
        ("State + Solver", ["attitude, orbit, wheel momentum;", "coupled RK4 / DOP853, or Keplerian / SGP4", "orbit with RK4 / RK45 attitude"], 106),
    ]
    y, parts, mids = 40, [], []
    for title, lines, h in inputs:
        parts.append(box(left_x, y, lw, h, title, lines))
        mids.append(y + h / 2)
        y += h + 14
    cx, cy, cw, ch = 500, 150, 330, 300
    core = f"""
  <rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="{t['accentSoft']}" stroke="{t['accent']}" stroke-width="1.6"/>
  <text x="{cx + cw / 2}" y="{cy + 44}" text-anchor="middle" font-family="{MONO}" font-size="20" font-weight="700" fill="{t['accent']}">simulate(…, backend=)</text>
  <text x="{cx + cw / 2}" y="{cy + 74}" text-anchor="middle" font-family="{SANS}" font-size="14" fill="{t['soft']}">same model, same API, three engines</text>"""
    for i, (name, sub) in enumerate((("Python", "Numba-compiled"), ("C++", "pybind11"), ("Rust", "PyO3"))):
        py = cy + 100 + i * 62
        core += f"""
  <rect x="{cx + 30}" y="{py}" width="{cw - 60}" height="48" rx="24" fill="{t['bg']}" stroke="{t['accent2']}"/>
  <text x="{cx + 58}" y="{py + 30}" font-family="{SANS}" font-size="17" font-weight="700" fill="{t['fg']}">{name}</text>
  <text x="{cx + cw - 58}" y="{py + 30}" text-anchor="end" font-family="{SANS}" font-size="13.5" fill="{t['soft']}">{sub}</text>"""
    arrows = "\n  ".join(arrow(left_x + lw, m, cx - 4, cy + 40 + i * 70) for i, m in enumerate(mids))
    rx, rw = 870, 378
    right = box(rx, 150, rw, 130, "SimulationResult",
                [f"{mono('ResultsFormat')} picks what is recorded:", "states, goals, sensor readings,",
                 "estimates, forces, torques, commands"], fill=t["card"])
    right += box(rx, 320, rw, 130, "Plotting",
                 [f"{mono('GenADCS.backends.shared.plotting')}", "attitude, pointing error, sensors,",
                  "estimator diagnostics and more"], fill=t["card"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="A GenADCS 3.0 simulation: a Satellite with its ADCS subsystem, an Environment, a DynamicsConfig, and an initial State with a Solver go into simulate(), which runs on the Python, C++ or Rust backend and returns a SimulationResult that the shared plotting module draws.">
  <rect width="{W}" height="{H}" rx="18" fill="{t['bg']}"/>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{t['line']}"/>
  {''.join(parts)}
  {core}
  {arrows}
  {arrow(cx + cw, cy + 120, rx - 4, 215, t['accent'])}
  {arrow(rx + rw / 2, 280, rx + rw / 2, 316)}
  {right}
</svg>
"""


def main():
    numbers = benchmark_numbers()
    data = mission_timeline()
    for name, t in THEMES.items():
        (OUT / f"hero-{name}.svg").write_text(hero(t, numbers))
        (OUT / f"architecture-{name}.svg").write_text(architecture(t))
        timeline(t, data, OUT / f"timeline-{name}.svg")
    print("wrote hero, architecture and timeline SVGs to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
