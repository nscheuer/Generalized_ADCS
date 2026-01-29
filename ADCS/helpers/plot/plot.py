__all__ = ["plot"]

import matplotlib.pyplot as plt

from ADCS.helpers.simresults import SimulationResults
from ADCS.helpers.simresults_mc import MCSimulationResults
from ADCS.helpers.plot.subplot import Subplot
from .close_all import ensure_close_all_button

def plot(
    sim: SimulationResults | MCSimulationResults,
    *subplots: Subplot,
    layout=None,
    figsize=(10, 8),
    title=None,
    sharex=True,
):
    r"""
    Render one or more subplot objects into a single matplotlib figure.

    This function provides a lightweight wrapper that arranges multiple
    :class:~ADCS.helpers.plot.subplot.Subplot instances into a shared figure
    using a specified layout. It is intended as the main user-facing entry
    point for generating simulation result plots.

    :param sim:
        Simulation results object supplying the data to all subplots.
    :type sim:
        :class:~ADCS.helpers.simresults.SimulationResults or :class:~ADCS.helpers.simresults_mc.MCSimulationResults

    :param subplots:
        One or more subplot objects to be rendered in the figure.
    :type subplots:
        :class:~ADCS.helpers.plot.subplot.Subplot

    :param layout:
        Tuple specifying the matplotlib subplot grid as (rows, columns).
        If None, a vertical layout is chosen automatically.
    :type layout:
        tuple[int, int] or None

    :param figsize:
        Size of the matplotlib figure in inches.
    :type figsize:
        tuple[float, float]

    :param title:
        Optional figure-level title.
    :type title:
        str or None

    :param sharex:
        If True, all subplots share the same x-axis.
    :type sharex:
        bool

    :return:
        None
    :rtype:
        None

    """
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
        
    plt.tight_layout()
    plt.show(block=False)
