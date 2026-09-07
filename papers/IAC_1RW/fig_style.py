"""Shared figure style for the IAC-26 paper. Serif to match the template,
Okabe-Ito palette, recessive axes/grid, consistent sizes."""
import matplotlib as mpl

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
      "grey": "#5A5A5A", "ink": "#222222"}


def apply(base=8.5):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 1.0,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.color": "#000000",
        "grid.alpha": 0.10,
        "grid.linewidth": 0.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#BBBBBB",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "lines.solid_capstyle": "round",
    })


def despine_keep(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
