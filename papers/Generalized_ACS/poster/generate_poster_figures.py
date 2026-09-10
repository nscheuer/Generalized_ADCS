"""Generate SSC26-P2-54 poster figures for Section IV.

The six vector PDF figures are written to ``poster/outputs``.  The
output location is based on this file, so the script can be run from any
working directory::

    python generate_poster_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCENT = "#A31F34"
INK = "#16181A"
MUTED = "#616A73"
RULE = "#C8CDD2"
WASH = "#F4F5F7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 20, "axes.linewidth": 2.0, "lines.linewidth": 3.5,
    "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def save(fig, name, **kwargs):
    fig.savefig(OUTPUT_DIR / name, **kwargs)
    plt.close(fig)


def slopegraph():
    rows = [("Lovera", 0, 29, "full"), ("Wisniewski", 2, 39, "full"),
            ("Lovera", 41, 80, "vector"), ("Wisniewski", 16, 72, "vector")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for name, a, b, task in rows:
        colour = ACCENT if task == "full" else MUTED
        ls = "-" if task == "full" else (0, (6, 3))
        ax.plot([0, 1], [a, b], color=colour, ls=ls, zorder=2,
                solid_capstyle="round")
        ax.plot([0, 1], [a, b], "o", color=colour, ms=11, zorder=3)
        dy = -12 if name == "Lovera" and task == "full" else 12 if task == "full" else 0
        ax.annotate(f"{a}%", (0, a), xytext=(-10, dy), textcoords="offset points",
                    ha="right", va="center", fontsize=19, color=colour)
        ax.annotate(f"{b}%  {name}", (1, b), xytext=(12, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=19, color=colour, weight="bold")
    ax.set_xlim(-.42, 1.72); ax.set_ylim(-8, 92)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["published\n(wheel idle)", "through the\nframework"], fontsize=20)
    ax.set_ylabel(r"trials converged below $5^\circ$", fontsize=20)
    ax.set_yticks([0, 20, 40, 60, 80]); ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.spines["left"].set_color(RULE)
    ax.tick_params(length=6, width=2); ax.grid(axis="y", color=RULE, lw=1, alpha=.6); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=ACCENT, lw=3.5, label="full attitude"),
                       Line2D([], [], color=MUTED, lw=3.5, ls=(0, (6, 3)), label="vector pointing")],
              loc="upper left", frameon=False, fontsize=19, handlelength=2.4)
    save(fig, "slopegraph.pdf")


def controllability():
    configs = ["1 MTQ", "2 MTQ", "3 MTQ", "3 MTQ + 1 RW", "3 MTQ + 2 RW", "3 MTQ + 3 RW"]
    cols = ["frozen", "orbit", "frozen", "orbit"]
    ok = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1],
                   [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    frac = np.array([["2/6", "4/6", "2/4", "2/4"], ["4/6", "6/6", "4/4", "4/4"],
                     ["4/6", "6/6", "4/4", "4/4"], ["6/6", "6/6", "4/4", "4/4"],
                     ["6/6", "6/6", "4/4", "4/4"], ["6/6", "6/6", "4/4", "4/4"]])
    nr, nc = ok.shape; fig, ax = plt.subplots(figsize=(7.4, 5.3))
    for i in range(nr):
        for j in range(nc):
            good = ok[i, j] == 1; y = nr - 1 - i
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=ACCENT if good else WASH,
                                       alpha=.16 if good else 1, edgecolor=RULE, lw=2))
            ax.text(j+.5, y+.60, "✓" if good else "✗", ha="center", va="center",
                    fontsize=30, color=ACCENT if good else MUTED, weight="bold")
            ax.text(j+.5, y+.24, frac[i, j], ha="center", va="center", fontsize=15, color=MUTED)
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.set_xticks(np.arange(nc)+.5); ax.set_xticklabels(cols, fontsize=16)
    ax.set_yticks(np.arange(nr)+.5); ax.set_yticklabels(configs[::-1], fontsize=18); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.text(1, nr+.40, "Full attitude", ha="center", va="bottom", fontsize=17, weight="bold")
    ax.text(3, nr+.40, "Vector pointing", ha="center", va="bottom", fontsize=17, weight="bold", color=INK)
    ax.plot([.08, 1.92], [nr+.32]*2, color=RULE, lw=2, clip_on=False); ax.plot([2.08, 3.92], [nr+.32]*2, color=RULE, lw=2, clip_on=False)
    ax.set_ylim(-.85, nr+.90); ax.add_patch(plt.Rectangle((0, nr-4), nc, 1, fill=False, edgecolor=ACCENT, lw=4, zorder=5))
    ax.text(nc, -.42, "boxed: the lightest bus controllable everywhere", fontsize=16, color=ACCENT, ha="right", va="center", weight="bold")
    save(fig, "controllability.pdf", transparent=True)


def envelope():
    inc = np.deg2rad(50); u = np.linspace(0, 2*np.pi, 160)
    B = np.vstack([np.cos(u)*np.sin(inc), -np.cos(inc)*np.ones_like(u), 2*np.sin(u)*np.sin(inc)])
    B /= np.linalg.norm(B, axis=0); theta = np.linspace(0, 2*np.pi, 220); cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(13.6, 5.0))
    def sphere(ax):
        a = np.linspace(0, 2*np.pi, 60); p = np.linspace(0, np.pi, 30)
        ax.plot_wireframe(np.outer(np.cos(a), np.sin(p)), np.outer(np.sin(a), np.sin(p)), np.outer(np.ones_like(a), np.cos(p)), color=RULE, lw=.5, alpha=.55, rstride=3, cstride=3)
        ax.set_box_aspect((1,1,1)); ax.set_xlim(-.8,.8); ax.set_ylim(-.8,.8); ax.set_zlim(-.8,.8); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off(); ax.view_init(elev=20, azim=38)
    def great_circle(n):
        n /= np.linalg.norm(n); a = np.array([0.,0.,1.]) if abs(n @ np.array([0.,0.,1.])) <= .9 else np.array([1.,0.,0.])
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        return (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)).T
    axL = fig.add_subplot(121, projection="3d"); sphere(axL)
    for k in range(0, B.shape[1], 7):
        c = great_circle(B[:, k]); axL.plot(c[0], c[1], c[2], color=cmap(k/B.shape[1]), lw=1.6, alpha=.85)
    axL.set_title(r"$\geq$ 2 magnetorquers\nswept planes cover all of $\mathbb{R}^3$", fontsize=21, color=INK, pad=-4)
    axR = fig.add_subplot(122, projection="3d"); sphere(axR); m = np.array([0.,0.,1.]); conf = great_circle(m)
    axR.plot(conf[0], conf[1], conf[2], color=MUTED, lw=2.6, alpha=.75)
    axR.text2D(0, .06, "torque confined\nto this plane", transform=axR.transAxes, color=MUTED, fontsize=17)
    for k in range(0, B.shape[1], 7):
        t = np.cross(m, B[:, k]); nrm = np.linalg.norm(t)
        if nrm < 1e-9: continue
        t /= nrm; axR.plot([0,t[0]], [0,t[1]], [0,t[2]], color=cmap(k/B.shape[1]), lw=2.8, alpha=.95); axR.plot([t[0]], [t[1]], [t[2]], "o", color=cmap(k/B.shape[1]), ms=5)
    axR.plot([0,0], [0,0], [-1.05,1.05], color=ACCENT, lw=4); axR.scatter([0,0], [0,0], [1.,-1.], color=ACCENT, s=110)
    axR.text(0, 0, 1.05, r"$\hat m$  never reachable", color=ACCENT, fontsize=18, ha="center", weight="bold")
    axR.set_title("one magnetorquer\ntorque always $\\perp\\ \\hat m$, rank capped", fontsize=21, color=INK, pad=-4)
    axL.set_position([-.06,.06,.60,.92]); axR.set_position([.46,.06,.60,.92]); cax = fig.add_axes([.37,.015,.26,.030])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1)), cax=cax, orientation="horizontal"); cb.set_ticks([0,1]); cb.set_ticklabels(["start of orbit", "one orbit"]); cb.ax.tick_params(labelsize=17, length=0); cb.outline.set_visible(False)
    save(fig, "envelope.pdf", bbox_inches=None)


def wheel_failure():
    cases = [("3 MTQ + 3 RW $\\to$ 2 RW\nfull LVLH held", None, 0., ACCENT), ("3 MTQ + 1 RW $\\to$ 0 RW\nboresight held", 13.5, 3., ACCENT), ("3 MTQ + 1 RW $\\to$ 0 RW\nroll released", None, 110., MUTED)]
    fig, ax = plt.subplots(figsize=(9., 4.1)); ys = np.arange(len(cases))[::-1]
    for y, (lab, peak, settled, col) in zip(ys, cases):
        if peak is not None:
            ax.hlines(y, settled, peak, color=col, lw=13, alpha=.25); ax.plot([peak], [y], "<", color=col, ms=15); ax.annotate(f"peak {peak:.1f}$^\\circ$", (peak,y), xytext=(14,0), textcoords="offset points", fontsize=17, color=col, va="center")
        ax.plot([settled], [y], "o", color=col, ms=15); off = (-14,0) if peak is not None else (14,0); ha = "right" if peak is not None else "left"
        ax.annotate(f"{settled:.0f}$^\\circ$", (settled,y), xytext=off, textcoords="offset points", fontsize=19, color=col, va="center", ha=ha, weight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in cases], fontsize=17); ax.set_ylim(-.6, len(cases)-.4); ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-.35,700); ax.set_xticks([0,1,10,100]); ax.set_xticklabels(["0","1","10","100"]); ax.set_xlabel("pointing error after one orbit [deg]", fontsize=18); ax.axvline(5, color=MUTED, ls=(0,(5,4)), lw=2); ax.annotate("$5^\\circ$", (5,len(cases)-.55), xytext=(4,0), textcoords="offset points", fontsize=16, color=MUTED)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.spines["left"].set_color(RULE); ax.tick_params(length=6, width=2, labelsize=18); ax.grid(axis="x", color=RULE, lw=1, alpha=.6); ax.set_axisbelow(True); save(fig, "wheelfail.pdf")


def goal_geometry():
    fig, ax = plt.subplots(figsize=(7.4,4.6)); ax.add_patch(plt.Circle((0,0),1, facecolor=WASH, edgecolor=RULE, lw=2.5)); cA,rA=np.array([-.30,.06]),.63; cB,rB=np.array([.29,-.11]),.61; d=np.linalg.norm(cB-cA); a=(rA*rA-rB*rB+d*d)/(2*d); h=np.sqrt(max(rA*rA-a*a,0)); p=cA+a*(cB-cA)/d; perp=np.array([-(cB-cA)[1],(cB-cA)[0]])/d; x1,x2=p+h*perp,p-h*perp; th=np.linspace(0,2*np.pi,300)
    ax.plot(cA[0]+rA*np.cos(th),cA[1]+rA*np.sin(th),color=ACCENT,lw=3.5); ax.plot(cB[0]+rB*np.cos(th),cB[1]+rB*np.sin(th),color=MUTED,lw=3,ls=(0,(7,4))); q=np.array([-1.32,.92]); v=q-cA; star=cA+rA*v/np.linalg.norm(v); ax.plot(*q,"o",color=INK,ms=13); ax.plot(*star,"o",color=ACCENT,ms=16); ax.plot([q[0],star[0]],[q[1],star[1]],color=INK,lw=2.2,ls=(0,(2,3))); ax.annotate("current $q$",q,xytext=(0,16),textcoords="offset points",fontsize=17,ha="center"); ax.annotate("$\\gamma^*$",star,xytext=(-4,14),textcoords="offset points",fontsize=20,color=ACCENT,ha="center",weight="bold")
    for xp in (x1,x2): ax.plot(*xp,"o",color=INK,ms=15)
    ax.annotate("$\\pm\\bar q_g$",x1,xytext=(16,6),textcoords="offset points",fontsize=19,color=INK,weight="bold"); ax.annotate("a vector goal is\none circle; the adapter\npicks a point on it",(.005,.30),xycoords="axes fraction",fontsize=16,color=ACCENT,ha="left",va="top",weight="bold"); ax.annotate("a full goal is two such\ncircles, meeting only\nat the target",(.995,.20),xycoords="axes fraction",fontsize=16,color=MUTED,ha="right",va="top"); ax.set_xlim(-2.05,1.95); ax.set_ylim(-1.35,1.35); ax.set_aspect("equal"); ax.axis("off"); save(fig,"goalgeom.pdf",transparent=True)


def alloc_geometry():
    fig, ax = plt.subplots(figsize=(7.4,4.6)); poly=np.array([[1.25,.16],[.60,.34],[-.60,.34],[-1.25,-.16],[-.60,-.34],[.60,-.34]]); ax.add_patch(plt.Polygon(poly,closed=True,facecolor=WASH,edgecolor=MUTED,lw=3)); ax.annotate(r"achievable torque set $\mathcal{T}(B_\tau)$",(0,-.52),fontsize=17,color=MUTED,ha="center"); req=np.array([.42,1.32]); ax.annotate("",xy=req,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",lw=3,color=INK,mutation_scale=26)); ax.annotate(r"requested $\tau_{comp}$",req,xytext=(4,8),textcoords="offset points",fontsize=18,ha="center")
    def boundary_hit(direction):
        best=None
        for i in range(len(poly)):
            p1,p2=poly[i],poly[(i+1)%len(poly)]; e=p2-p1; M=np.array([[direction[0],-e[0]],[direction[1],-e[1]]])
            if abs(np.linalg.det(M))<1e-9: continue
            t,u=np.linalg.solve(M,p1)
            if t>0 and -1e-9<=u<=1+1e-9: best=t if best is None else min(best,t)
        return direction*best
    lp=boundary_hit(req/np.linalg.norm(req)); ax.annotate("",xy=lp,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",lw=5,color=ACCENT,mutation_scale=24)); ax.annotate("LP: direction held,\nmagnitude scaled to fit",(-.30,.86),fontsize=17,color=ACCENT,ha="right",weight="bold")
    best,bd=None,1e9
    for i in range(len(poly)):
        p1,p2=poly[i],poly[(i+1)%len(poly)]; e=p2-p1; t=np.clip((req-p1)@e/(e@e),0,1); c=p1+t*e
        if np.linalg.norm(req-c)<bd: bd,best=np.linalg.norm(req-c),c
    ax.annotate("",xy=best,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",lw=3.5,color=MUTED,linestyle=(0,(5,3)),mutation_scale=22)); ax.annotate("QP: magnitude kept,\ndirection tilts",(1.05,.72),fontsize=17,color=MUTED,ha="left",weight="bold"); aL=np.arctan2(lp[1],lp[0]); aQ=np.arctan2(best[1],best[0]); arc=np.linspace(aQ,aL,60); r=.62; ax.plot(r*np.cos(arc),r*np.sin(arc),color=INK,lw=2.4); mid=(aL+aQ)/2; ax.annotate("direction\nerror",(r*np.cos(mid)*1.55,r*np.sin(mid)*1.42),fontsize=16,color=INK,ha="center",va="center"); ax.plot(0,0,"o",color=INK,ms=9); ax.set_xlim(-2.15,2.35); ax.set_ylim(-.85,1.75); ax.set_aspect("equal"); ax.axis("off"); save(fig,"allocgeom.pdf",transparent=True)


if __name__ == "__main__":
    slopegraph(); controllability(); envelope(); wheel_failure(); goal_geometry(); alloc_geometry()
    for path in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"{path.name} {path.stat().st_size} bytes")
