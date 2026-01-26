import numpy as np
from ..subplot import Subplot

class QuaternionPlotCombined(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Quaternion Components",
        units: str = "",
        colors=("tab:blue", "tab:orange", "tab:green", "tab:red"),
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        X = np.vstack(sim.state_hist)
        q = X[:, 3:7]

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

        for i in range(4):
            ax.plot(
                t,
                q[:, i],
                color=self.colors[i],
                label=labels[i]
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Quaternion {self.units}".strip())
        ax.set_title(self.title)
        ax.legend()
        ax.grid(True)
