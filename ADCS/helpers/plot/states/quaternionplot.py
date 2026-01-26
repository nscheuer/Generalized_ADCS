import numpy as np
import matplotlib.gridspec as gridspec
from ..subplot import Subplot

class QuaternionPlot(Subplot):
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
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False,
                       bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=ax.get_subplotspec()
        )

        axes = [
            ax.figure.add_subplot(gs[0, 0]),  # q0
            ax.figure.add_subplot(gs[0, 1]),  # q1
            ax.figure.add_subplot(gs[1, 0]),  # q2
            ax.figure.add_subplot(gs[1, 1]),  # q3
        ]

        t = getattr(sim, self.time)
        X = np.vstack(sim.state_hist)
        q = X[:, 3:7]

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

        for i, ax_i in enumerate(axes):
            ax_i.plot(t, q[:, i], color=self.colors[i], label=labels[i])
            ax_i.set_ylabel(f"{labels[i]} {self.units}".strip())
            ax_i.legend()
            ax_i.grid(True)

        axes[2].set_xlabel("Time [s]")
        axes[3].set_xlabel("Time [s]")

        axes[0].set_title(self.title, loc="left", pad=10)
