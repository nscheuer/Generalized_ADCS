"""House style for the IAC-26 paper figures.

One font family (serif/STIX to match the template), labels at body-text size
(10 pt) or larger, few ticks, no full frame, thin gridlines only where they help.

Colour grammar (fixed across the paper):
  architecture -> colour   : 3+0 vermilion, 3+1 blue, 3+3 green
  controller   -> line/mark: PD solid / circle, planner dashed / triangle
Figure 1's mountings are a different dimension: line style, one ink colour.
"""
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

ARCH = {"3+0": "#D55E00", "3+1": "#0072B2", "3+3": "#009E73"}
CTRL = {"PD": dict(ls="-", marker="o"), "planner": dict(ls="--", marker="^")}
INK = "#222222"
MUTED = "#6E6E6E"
BAND_RESTORE = "#56B4E9"   # light blue zone
BAND_DUMP = "#E69F00"      # light orange zone
BAND_RANKLOSS = "#333333"  # thin dark strip


def apply(base=10.0):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "grid.color": "#000000",
        "grid.alpha": 0.08,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "lines.solid_capstyle": "round",
    })


def few_ticks(ax, nx=5, ny=5):
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))


def log_decades_only(ax, axis="both"):
    for a in ((ax.xaxis, ax.yaxis) if axis == "both" else
              ((ax.xaxis,) if axis == "x" else (ax.yaxis,))):
        a.set_major_locator(LogLocator(base=10, numticks=6))
        a.set_minor_formatter(NullFormatter())
