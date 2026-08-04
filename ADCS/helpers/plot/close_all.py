import matplotlib.pyplot as plt
from matplotlib.widgets import Button

_close_window_created = False
_close_button_ref = None

def ensure_close_all_button():
    global _close_window_created, _close_button_ref
    
    if _close_window_created:
        return

    fig = plt.figure(figsize=(3, 2))
    fig.canvas.manager.set_window_title("Control")
    fig.suptitle("Close All Figures", fontsize=12)

    ax_button = fig.add_axes([0.25, 0.35, 0.5, 0.3])
    
    # Create the button
    btn = Button(ax_button, "CLOSE ALL", color="red", hovercolor="darkred")

    def close_all(event):
        plt.close("all")
        global _close_window_created, _close_button_ref
        _close_window_created = False
        _close_button_ref = None

    btn.on_clicked(close_all)

    _close_button_ref = btn 

    plt.show(block=False)
    _close_window_created = True