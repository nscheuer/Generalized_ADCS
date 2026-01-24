import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def create_close_all_button_window() -> None:
    r"""
    Create a small control window with a button that closes all Matplotlib figures.

    This function opens a dedicated Matplotlib figure containing a single
    prominently colored button. When pressed, the button closes **all**
    currently open Matplotlib figures, including those with active animations
    or interactive callbacks.

    The function is useful as a global emergency stop for visualization-heavy
    workflows.

    :param:
        None.
    :type:
        None

    :return:
        None. All open Matplotlib figures are closed upon user interaction.
    :rtype:
        None

    """

    # Create a small dedicated figure
    fig = plt.figure(figsize=(3, 2))
    fig.suptitle("Close All Figures", fontsize=12)

    # A centered button covering most of the figure
    ax_button = fig.add_axes([0.25, 0.35, 0.5, 0.3])  # [left, bottom, width, height]
    btn = Button(ax_button, "CLOSE ALL", color="red", hovercolor="darkred")

    # Action callback
    def close_all(event):
        plt.close("all")

    btn.on_clicked(close_all)
    plt.show()