import numpy as np
from ..subplot import Subplot

class QuaternionPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: int,          # 0, 1, 2, or 3
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        color: str | None = None,
    ):
        if component not in {0, 1, 2, 3}:
            raise ValueError("component must be an integer: 0, 1, 2, or 3")

        self.component = component
        self.time = time
        self.units = units
        self.color = color
        self.title = title

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        X = np.vstack(sim.state_hist)
        q = X[:, 3:7]

        y = q[:, self.component]
        
        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
        default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        
        label = labels[self.component]
        color = self.color or default_colors[self.component]
        title = self.title or f"{label} vs Time"

        ax.plot(t, y, color=color, label=label)
        
        ylabel = f"{label} [{self.units}]" if self.units else label
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both")