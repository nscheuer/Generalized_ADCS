"""Registered wave reads, adjudicated exactly as written.

A. offset/authority class split (PD-full kp1 vs kp2, paired)
B. route classification, reduced (allocator-state criterion; screen alignment;
   misses 15/53)
C. demand-ratio recomputation (omega_perp, directional instantaneous authority)
D. screen sensitivity, sigma-threshold axis
E. cross-controller hardness overlap (hypergeometric, 0.01 inference line)
F. aging paired deltas (cell E vs cell 2) -- ledger
"""
import glob
import os
import pickle
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import (  # noqa: E402
    _get_orbit, error_series, EPOCH, IAC_6U)
from ADCS.orbits.universal_constants import TimeConstants  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
S2C = TimeConstants.sec2cent
KD = 8.683e-3
M_MAX = 0.6


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, "rb") as f:
            r = pickle.load(f)
        out[int(r["config"]["seed"])] = r
    return out


def main():
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    k1 = load(os.path.join(OUT, "wave", "pd_full_kp1", "*.pkl"))
    k2 = load(os.path.join(OUT, "wave", "pd_full_kp2", "*.pkl"))
    red = load(os.path.join(OUT, "wave", "pd_reduced_kp1", "*.pkl"))
    pf = load(os.path.join(OUT, "tune_seed*_wave_planner_full.pkl"))
    pe = load(os.path.join(OUT, "tune_seed*_wave_planner_full_base.pkl"))
    c2 = load(os.path.join(OUT, "A_trials", "1rw_full_planner_seed*.pkl"))

    # ---------- A. class split ----------
    say("== A. OFFSET/AUTHORITY SPLIT (registered rule, paired PD-full) ==")
    auth, offs, resid = [], [], []
    for s in range(100):
        e1 = float(error_series(k1[s])[-1]); e2 = float(error_series(k2[s])[-1])
        h1 = float(np.max(k1[s]["h_frac"])); h2 = float(np.max(k2[s]["h_frac"]))
        if e1 <= 5.0 and e2 <= 5.0:
            continue
        if h1 >= 0.999 or h2 >= 0.999 or e2 > 30.0:
            auth.append(s)
        elif e1 > 5.0 and e2 < 0.75 * e1:
            offs.append(s)
        else:
            resid.append(s)
    say(f"failures at kp1: {sum(float(error_series(k1[s])[-1])>5 for s in range(100))}; "
        f"at kp2: {sum(float(error_series(k2[s])[-1])>5 for s in range(100))}")
    say(f"AUTHORITY-class: {len(auth)} {auth}")
    say(f"OFFSET-class: {len(offs)} {offs}")
    say(f"unclassified residual: {len(resid)} {resid}")

    # ---------- B. route classification, reduced ----------
    say("\n== B. ROUTE CLASSIFICATION, reduced (allocator-state criterion) ==")
    div, starved, catches = [], [], []
    miss_info = {}
    for s, r in red.items():
        e = error_series(r)
        if float(e[-1]) <= 30.0:
            continue
        div.append(s)
        u = np.asarray(r["u"], float)
        h = np.asarray(r["state"], float)[:, 7]
        n_mtq = int(r["n_mtq"])
        uw = u[:, n_mtq]
        despin = (uw[:len(h)] * h[:len(u)]) > 0
        mtq_bind = np.max(np.abs(u[:, :n_mtq]), axis=1) >= 0.999 * M_MAX
        dfrac = float(np.mean(despin))
        bind_given_despin = float(np.mean(mtq_bind[despin])) if despin.any() else 0.0
        is_starved = dfrac >= 0.30 and bind_given_despin >= 0.90
        sg = np.asarray(r["sigma"], float)
        dwell = float(np.mean(sg[np.isfinite(sg)] < 0.2))
        if is_starved:
            starved.append(s)
        if dwell <= 0.1035:
            catches.append(s)
        miss_info[s] = (round(dfrac, 2), round(bind_given_despin, 3), round(dwell, 4),
                        is_starved, dwell <= 0.1035)
    say(f"diverged: {len(div)} {sorted(div)}")
    say("seed: (despin_frac, bind|despin, dwell, starved?, flagged?)")
    for s in sorted(div):
        say(f"  {s}: {miss_info[s]}")
    st, ca = set(starved), set(catches)
    say(f"STARVED (independent criterion): {sorted(st)}")
    say(f"screen catches: {sorted(ca)}")
    say(f"alignment: catches&starved {len(ca & st)}, catches-only {len(ca - st)}, "
        f"starved-missed {len(st - ca)}")
    say("mechanism-conditional recall RETURNS only if catches ~= starved set.")

    # ---------- C. demand ratio ----------
    say("\n== C. DEMAND RATIO recomputed (omega_perp, directional instantaneous authority) ==")
    conv_pt, div_pt, conv_pk, div_pk = [], [], [], []
    for s, r in red.items():
        e = error_series(r)
        t = np.asarray(r["time"], float)
        st_ = np.asarray(r["state"], float)
        om = st_[:, 0:3]
        h = st_[:, 7]
        om_perp = np.linalg.norm(om[:, :2], axis=1)          # wheel axis = body z
        cfg = r["config"]
        orb = _get_orbit(cfg, 1.0, float(cfg["tf"]))
        idx = np.arange(0, len(t), 60)
        Bm = np.array([np.linalg.norm(np.asarray(
            orb.get_os(J2000=EPOCH + t[i] * S2C).B, float)) for i in idx])
        num = np.sqrt((h[idx] * IAC_6U.h_max) ** 2 + KD ** 2) * om_perp[idx]
        den = M_MAX * Bm
        ratio = num / den
        post = idx > 1000
        pk = float(np.max(ratio)); pt = float(np.median(ratio[post])) if post.any() else np.nan
        (div_pk if float(e[-1]) > 30 else conv_pk).append(pk)
        (div_pt if float(e[-1]) > 30 else conv_pt).append(pt)
    say(f"converged: post-transient median {np.median(conv_pt):.2f} "
        f"(IQR {np.percentile(conv_pt,25):.2f}-{np.percentile(conv_pt,75):.2f}); "
        f"peak median {np.median(conv_pk):.2f}")
    say(f"diverged:  post-transient median {np.median(div_pt):.2f}; "
        f"peak median {np.median(div_pk):.2f}")
    say("registered: converged near unity => validated threshold; materially above =>"
        " separator only + residual named.")

    # ---------- D. sigma-threshold axis ----------
    say("\n== D. SCREEN sigma-threshold axis (rerun labels) ==")
    y = {s: float(error_series(r)[-1]) > 30 for s, r in red.items()}
    for sth in (0.15, 0.2, 0.25):
        dw = {s: float(np.mean(np.asarray(r["sigma"], float)[
            np.isfinite(np.asarray(r["sigma"], float))] < sth)) for s, r in red.items()}
        row = []
        for cut in (0.075, 0.1035, 0.125):
            c = sum(1 for s in red if dw[s] <= cut and y[s])
            fp = sum(1 for s in red if dw[s] <= cut and not y[s])
            row.append(f"cut {cut}: {c}/{sum(y.values())} FP {fp}")
        say(f"  sigma<{sth}: " + " | ".join(row))

    # ---------- E. hardness overlap ----------
    say("\n== E. CROSS-CONTROLLER HARDNESS OVERLAP (registered) ==")
    K = sorted(s for s, r in pf.items() if r.get("n_budget_kills", 0))
    A = sorted(auth)
    ov = sorted(set(K) & set(A))
    say(f"|K|={len(K)} kill-seeds; |A|={len(A)} authority-class; overlap {len(ov)}: {ov}")
    N, kk, aa, x = 100, len(K), len(A), len(ov)
    p = sum(comb(aa, i) * comb(N - aa, kk - i) for i in range(x, min(kk, aa) + 1)) / comb(N, kk)
    say(f"hypergeometric P(overlap >= {x}) = {p:.4f}  (inference line 0.01; "
        f"E[overlap] = {kk*aa/100:.1f})")

    # ---------- F. aging deltas (ledger) ----------
    say("\n== F. AGING PAIRED DELTAS (ledger) ==")
    ds = []
    for s in range(100):
        if s in pe and s in c2 and not c2[s].get("n_budget_kills", 0):
            ds.append(abs(float(error_series(pe[s])[-1]) - float(error_series(c2[s])[-1])))
    say(f"paired |delta final| over {len(ds)} clean-comparable seeds: median "
        f"{np.median(ds):.3f} deg, p90 {np.percentile(ds,90):.3f}, max {np.max(ds):.2f}")

    with open(os.path.join(OUT, "WAVE_READS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    say("\nwritten: output_data/WAVE_READS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
