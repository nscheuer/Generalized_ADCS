"""Shared publication style for the Paper-1 / Paper-2 figures.

Single source of truth for the IEEE-conference look: serif (Times/STIX) fonts,
consistent sizes, a single colour palette, and column-width helpers. Import and
call `use_pub_style()` once at the top of a figure script, size figures with
`COL_W` / `DBL_W`, and pull colours from `PALETTE` so every figure matches.

    from papers._pubstyle import use_pub_style, COL_W, DBL_W, PALETTE
    use_pub_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.75))
    ax.plot(t, y, color=PALETTE["planner"])

Falls back silently to mathtext if a serif system font isn't found, so it is
safe on headless machines.
"""
from __future__ import annotations

import matplotlib as mpl

# IEEE two-column: text column ~3.5 in, full text width ~7.16 in.
COL_W = 3.5      # single-column figure width [in]
DBL_W = 7.16     # double-column (full-width) figure width [in]

# One palette for every figure. "planner" vs "pd" are the two recurring roles;
# the rest are for multi-trace panels (rate/momentum/commands) and disturbances.
PALETTE = {
    "planner": "#2c7fb8",   # blue  -- ALTRO / SALTRO planner
    "pd":      "#de2d26",    # red   -- PD / LP / Lovera baseline
    "accent":  "#31a354",    # green -- third trace (e.g. body rate)
    "warn":    "#e6a000",    # amber -- disturbance / event
    "neutral": "#636363",    # grey  -- reference / annotation
    "full":    "#2c7fb8",    # full-attitude (3-DOF) error
    "bore":    "#e6a000",    # boresight (2-DOF) error
}

# distinct colours for the x/y/z traces of a 3-vector (commands, momentum, rate)
AXIS_COLORS = ("#1b9e77", "#d95f02", "#7570b3")


def use_pub_style() -> None:
    """Apply the shared rcParams. Idempotent; call once per script."""
    mpl.rcParams.update({
        # --- fonts: serif body, matching math ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        # --- lines / ticks ---
        "lines.linewidth": 1.3,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # --- grid / spines ---
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # --- legend ---
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        # --- output ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (editable text in the PDF)
        "ps.fonttype": 42,
    })
