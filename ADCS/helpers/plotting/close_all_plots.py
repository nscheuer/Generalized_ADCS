import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def create_close_all_button_window() -> None:
    """
    Create a small figure with a single red button centered.
    Pressing the button closes ALL open Matplotlib figures
    (including any active animations).
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