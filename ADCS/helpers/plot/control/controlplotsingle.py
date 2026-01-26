import numpy as np
import matplotlib.pyplot as plt
from ..subplot import Subplot

class ControlPlotSingle(Subplot):
    def __init__(
        self,
        index: int,
        *,
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        label: str | None = None,
        color: str | None = None,
        log_y: bool = False,
    ):
        self.index = index
        self.time = time
        self.title = title
        self.units = units
        self.label = label
        self.color = color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        if sim.control_hist is None or len(sim.control_hist) == 0:
            self._plot_no_data(ax)
            return

        t = getattr(sim, self.time)
        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        if self.index < 0 or self.index >= n_ctrl:
            raise ValueError(f"Control index {self.index} out of bounds for {n_ctrl} channels.")

        y = U[:, self.index]
        
        if self.label:
            lbl = self.label
        else:
            lbl = rf"$u_{{{self.index}}}$"

        if self.title:
            tit = self.title
        else:
            tit = f"Control Channel {self.index}"

        c = self.color if self.color else "tab:blue"

        ax.plot(t, y, color=c, label=lbl)
        
        ylabel = f"{lbl} [{self.units}]" if self.units else lbl
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(tit)
        ax.legend()
        ax.grid(True, which="both")

        if self.log_y:
            ax.set_yscale("log")

    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title or "Control Input", loc="left", pad=10)
        ax.text(0.5, 0.5, "No control history available", ha="center", va="center", transform=ax.transAxes)