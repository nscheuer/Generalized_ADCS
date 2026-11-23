"""A small Matplotlib window for closing all open figures."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


_control_figure = None


def ensure_close_all_button():
    """Open one control window while plots are being created."""
    global _control_figure

    if _control_figure is not None and plt.fignum_exists(_control_figure.number):
        return

    fig = plt.figure(figsize=(3, 2))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Control")
    fig.suptitle("Close All Figures", fontsize=12)
    ax_button = fig.add_axes([0.25, 0.35, 0.5, 0.3])
    button = Button(ax_button, "CLOSE ALL", color="red", hovercolor="darkred")

    def close_all(_event):
        global _control_figure
        plt.close("all")
        _control_figure = None

    button.on_clicked(close_all)
    fig._close_all_button = button  # Keep the widget alive for as long as the figure is open.
    _control_figure = fig
    plt.show(block=False)
