"""Shared publication style for the IAC_1RW paper figures."""

import matplotlib as mpl


BLUE = "#0072B2"       # MTQ / resultant authority
RED = "#D55E00"        # RW line / direction vector
PURPLE = "#7B3294"     # magnetic dipole
ORANGE = "#E69F00"     # RW cube
GRID = "#B8B8B8"


def configure_ieee_style():
    """Use compact, legible, color IEEE-style defaults for paper figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.6,
        "grid.linewidth": 0.45,
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
