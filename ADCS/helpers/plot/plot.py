import matplotlib.pyplot as plt

from ADCS.helpers.simresults import SimulationResults
from ADCS.helpers.plot.subplot import Subplot
from .close_all import ensure_close_all_button

def plot(
    sim: SimulationResults,
    *subplots: Subplot,
    layout=None,
    figsize=(10, 8),
    title=None,
    sharex=True,
):
    ensure_close_all_button()

    n = len(subplots)
    if layout is None:
        layout = (n, 1)

    fig, axes = plt.subplots(*layout, figsize=figsize, sharex=sharex)
    fig.canvas.manager.set_window_title(title)

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if len(axes) < n:
        raise ValueError("Layout too small for number of subplots")

    for ax, subplot in zip(axes, subplots):
        subplot.plot(ax, sim)

    if title:
        fig.suptitle(title)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show(block=False)
