import numpy as np
import matplotlib.pyplot as plt
from ..subplot import Subplot

class ControlPlotCombined(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        colors: list[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.colors = colors

    def plot(self, ax, sim) -> None:
        if sim.control_hist is None or len(sim.control_hist) == 0:
            self._plot_no_data(ax)
            return

        t = getattr(sim, self.time)
        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        if self.labels is None:
            labels = [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
        else:
            if len(self.labels) != n_ctrl:
                raise ValueError(f"Label count ({len(self.labels)}) does not match control channels ({n_ctrl})")
            labels = self.labels

        for i in range(n_ctrl):
            color_arg = {}
            if self.colors:
                color_arg['color'] = self.colors[i % len(self.colors)]
            
            ax.plot(t, U[:, i], label=labels[i], **color_arg)

        ylabel = f"Control Input [{self.units}]" if self.units else "Control Input"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(self.title)
        
        if self.log_y:
            ax.set_yscale("log")
            
        if n_ctrl > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend()
            
        ax.grid(True, which="both", linestyle='--', alpha=0.7)

    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(0.5, 0.5, "No control history available", ha="center", va="center", transform=ax.transAxes)