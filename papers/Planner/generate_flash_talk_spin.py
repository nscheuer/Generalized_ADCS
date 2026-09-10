#!/usr/bin/env python
"""
Flash-talk slide 3 -- the two stage-weight panels of the emergent-spin demo.

Same single open-loop SALTRO solve as P2.2 (`generate_p2.2_spin.py`, whose
scenario/settings this file imports verbatim rather than re-declaring): oblate
bus J = diag(0.10, 0.05, 0.005) kg*m^2, 3 MTQ + 1 RW on +y_b (h_max = 2
mN*m*s), constant body-fixed tau_d = 0.3 mN*m on +x_b, 6800 km / 51.5 deg,
500 s horizon, roll-free anti-velocity goal, at rest 179 deg off target.

Fig 6a/6b restyled for the deck -- vector PDF, no titles, no legends, direct
line-end labels.

OUTPUT
------
papers/Planner/flash_talk/figs/spin_pointing.pdf   (Panel A)
papers/Planner/flash_talk/figs/spin_rates.pdf      (Panel B)
papers/Planner/output_data/spin_traj_flashtalk.npz (the solved trajectory)

The solve is the slow part, so it is cached: `--from-npz` re-renders both
panels from the dump, which is what you want when iterating on style.

SALTRO BUILD REQUIREMENT -- read this before re-running the solve
-----------------------------------------------------------------
No saltro_py build sitting on this machine can run it, and the repo's own
`SALTRO/build` is not close. Two knobs have to coexist:

  * the wrapper (`SALTRO_pass_settings.to_cpp`) sets `cpp_cost.RWh_max_mult`
    UNGUARDED, and SALTRO's knee re-key (9145b80, 2026-06-10) deleted it;
  * this scenario needs `rw_momentum_limit_scale` (#39) and
    `ang_vel_roll_ratio` (#26), which pre-date nothing older than 2026-05-29.

So every pre-re-key build fails on #39 and every post-re-key build fails on
`RWh_max_mult`. The one commit carrying all three is **6d9f8f0**
("checkpoint(autonomous-spin)", 2026-05-29, on `origin/autonomous-spin-backup`)
-- the commit the canonical run was made from. Build it in a worktree outside
~/Documents:

  git -C ~/Documents/SALTRO worktree add --detach /tmp/saltro-spin-canon 6d9f8f0
  cd /tmp/saltro-spin-canon
  VENV=~/Documents/Generalized_ADCS/venv
  VIRTUAL_ENV=$VENV PATH=$VENV/bin:$PATH $VENV/bin/cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR=$VENV \
      -DPython_FIND_VIRTUALENV=ONLY -DPYBIND11_FINDPYTHON=ON \
      -DSALTRO_WARNINGS_AS_ERRORS=OFF
  $VENV/bin/cmake --build build --target saltro_py -j8

Run (from the repo root):
  ./venv/bin/python papers/Planner/generate_flash_talk_spin.py \
      --saltro-build /tmp/saltro-spin-canon/build

Verified 2026-08-20 on that build: ok=True, 1 deg crossing 151 s, PE_fin
0.121 deg, |h|_max 1.70 mN*m*s (exact match to the canonical run), settled
<w_z> 13.77 deg/s.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- deck style --------------------------------------------------------------
PRIMARY = "#993333"
SECONDARY = "#9E9E9E"
ANNOT = "#616161"
FIGSIZE = (4.2, 2.35)          # 62 x 34.5 mm slots
LW = 2.5

STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.size": 12,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


def load_p22():
    """Import `generate_p2.2_spin.py` -- the dots make it un-importable by name."""
    path = os.path.join(HERE, "generate_p2.2_spin.py")
    spec = importlib.util.spec_from_file_location("p22_spin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve(saltro_build):
    """Run the P2.2 solve and return the trajectory in plot units."""
    p22 = load_p22()
    sys.path.append(os.path.expanduser(saltro_build))
    import saltro_py as S
    from ADCS.helpers.math_helpers import rot_mat

    sat = p22.build_satellite()
    ps = p22.make_settings(sat)
    jt, qg, bs, r0, v0, n = p22.grids(p22.DT)
    x0 = np.concatenate([np.zeros(3), p22.Q0_THESIS, [0.0]])

    ok, X, U, _K = S.trajOpt(ps.to_cpp(), p22.cs_satellite(S), x0, r0, v0,
                             np.ascontiguousarray(jt), np.ascontiguousarray(qg),
                             np.ascontiguousarray(bs))
    X = np.asarray(X); U = np.asarray(U)

    t = (jt - jt[0]) * 36525.0 * 86400.0
    pe = np.array([float(np.degrees(np.arccos(np.clip(
        (rot_mat(X[3:7, k]) @ np.array([0, 0, 1]))
        @ (qg[1:4, k] / np.linalg.norm(qg[1:4, k])), -1, 1))))
        for k in range(X.shape[1])])
    w = X[0:3, :] * 180.0 / np.pi
    h = X[7, :] * 1000.0
    return bool(ok), t, pe, w, h, U


SETTLED_FROM = 300.0   # s; steady state, well clear of the 151 s acquisition


def metrics(t, pe, w, pe_metric="mean"):
    """The three numbers the panels annotate, measured from this run.

    The settled window (t >= SETTLED_FROM) is what makes <w_z> come out at the
    published 13.8 deg/s; a last-30-samples mean sits on a local wobble and
    reads 13.98. `pe_metric="final"` switches Panel A's tail number to PE_fin
    (0.12 deg here), which is the metric the P2.2 header and the figure brief
    quote; "mean" is the settled-window mean (0.11 deg).
    """
    below = np.flatnonzero(pe < 1.0)
    t_acq = float(t[below[0]]) if below.size else float("nan")
    settled = t >= SETTLED_FROM
    pe_tail = float(pe[-1]) if pe_metric == "final" else float(np.mean(pe[settled]))
    wz_tail = float(np.mean(w[2][settled]))
    return t_acq, pe_tail, wz_tail


def panel_pointing(t, pe, t_acq, pe_tail, out, pe_word="mean"):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(t, pe, color=PRIMARY, lw=LW, solid_capstyle="round")

        ax.set_xlabel("time [s]")
        # two lines: at 14 pt a single rotated line is taller than the 34.5 mm
        # slot's axes and gets clipped
        ax.set_ylabel("pointing\nerror [deg]", linespacing=0.95)
        ax.set_xlim(0, 500)
        ax.set_ylim(-8, 195)
        ax.set_yticks([0, 60, 120, 180])
        ax.grid(True, color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # arrow at the 1-deg crossing; the text sits in the empty upper right,
        # clear of the descent (which sweeps 180 -> 0 between t ~ 20 and 150)
        ax.annotate(f"acquired {t_acq:.0f} s",
                    xy=(t_acq, 1.0), xycoords="data",
                    xytext=(238, 112), textcoords="data",
                    color=ANNOT, fontsize=11, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=ANNOT, lw=1.2,
                                    shrinkA=0, shrinkB=2))
        # tail note. Under the "final" metric the qualifier is dropped rather
        # than set to "final": "holds 0.12°" is both true and better stage
        # weight, and calling PE_fin a "mean" on a slide would be wrong.
        tail = f"holds {pe_tail:.2f}°" + (f" {pe_word}" if pe_word else "")
        ax.text(495, 26, tail,
                color=ANNOT, fontsize=11, ha="right", va="bottom")

        fig.tight_layout(pad=0.45)
        fig.savefig(out)
        plt.close(fig)


def panel_rates(t, w, wz_tail, out):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(wz_tail, ls="--", lw=0.9, color=SECONDARY, alpha=0.6, zorder=1)
        ax.plot(t, w[0], color=SECONDARY, lw=LW, solid_capstyle="round", zorder=2)
        ax.plot(t, w[1], color=SECONDARY, lw=LW, solid_capstyle="round", zorder=2)
        ax.plot(t, w[2], color=PRIMARY, lw=LW, solid_capstyle="round", zorder=3)

        ax.set_xlabel("time [s]")
        ax.set_ylabel("body rate\n[deg/s]", linespacing=0.95)
        ax.set_xlim(0, 500)
        ax.grid(True, color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        lo = float(min(w.min(), 0.0))
        hi = float(max(w.max(), wz_tail))
        pad = 0.14 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 2.0 * pad)

        # direct line-end labels, kept inside the 0-500 s axis
        ax.text(495, wz_tail + 0.9 * pad,
                f"{wz_tail:.1f}°/s — never commanded",
                color=PRIMARY, fontsize=11, ha="right", va="bottom")
        ax.text(495, float(w[0, -1]) - 0.5 * pad, "→ 0",
                color=ANNOT, fontsize=11, ha="right", va="top")

        fig.tight_layout(pad=0.45)
        fig.savefig(out)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saltro-build",
                    default="/tmp/saltro-spin-canon/build",
                    help="dir containing a saltro_py built from SALTRO 6d9f8f0 "
                         "(see the module docstring -- no existing build works)")
    ap.add_argument("--from-npz", action="store_true",
                    help="skip the solve; re-render both panels from the dump")
    ap.add_argument("--pe-metric", choices=["mean", "final"], default="final",
                    help="Panel A tail number: PE_fin (0.12 deg, the number the "
                         "paper and the figure brief quote -- default) or the "
                         "settled-window mean (0.11 deg)")
    ap.add_argument("--preview", action="store_true",
                    help="also write PNG copies next to the PDFs")
    args = ap.parse_args()

    npz_path = os.path.join(HERE, "output_data", "spin_traj_flashtalk.npz")
    fig_dir = os.path.join(HERE, "flash_talk", "figs")
    os.makedirs(fig_dir, exist_ok=True)

    if args.from_npz:
        d = np.load(npz_path)
        t, pe, w = d["t"], d["pe"], d["w"]
    else:
        ok, t, pe, w, h, U = solve(args.saltro_build)
        print(f"trajOpt ok = {ok}")
        if not ok:
            print("WARNING: trajOpt did not converge; not writing figures.")
            return 1
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez_compressed(npz_path, t=t, pe=pe, w=w, h=h, U=U)
        print(f"wrote {npz_path}")

    t_acq, pe_tail, wz_tail = metrics(t, pe, w, args.pe_metric)
    pe_word = "" if args.pe_metric == "final" else "mean"
    print(f"measured:  acquired (1 deg crossing) = {t_acq:.0f} s   "
          f"PE {pe_word} = {pe_tail:.3f} deg   "
          f"settled <w_z> (t>={SETTLED_FROM:.0f}s) = {wz_tail:.2f} deg/s   "
          f"PE_0 = {pe[0]:.1f} deg")

    for ext in (("pdf", "png") if args.preview else ("pdf",)):
        panel_pointing(t, pe, t_acq, pe_tail,
                       os.path.join(fig_dir, f"spin_pointing.{ext}"), pe_word)
        panel_rates(t, w, wz_tail, os.path.join(fig_dir, f"spin_rates.{ext}"))
    print(f"wrote spin_pointing.pdf and spin_rates.pdf to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
