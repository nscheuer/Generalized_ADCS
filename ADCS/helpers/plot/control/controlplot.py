import math
import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot

class ControlPlot(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        if sim.control_hist is None or len(sim.control_hist) == 0:
            fig = ax.figure
            ax_text = fig.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No control history available", ha="center", va="center")
            return

        t = getattr(sim, self.time)

        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        ncols = int(math.ceil(math.sqrt(n_ctrl)))
        nrows = int(math.ceil(n_ctrl / ncols))

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax.get_subplotspec())

        axes = []
        for i in range(n_ctrl):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        if self.labels is None:
            labels = [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
        else:
            if len(self.labels) != n_ctrl:
                raise ValueError(f"labels length ({len(self.labels)}) must match number of controls ({n_ctrl})")
            labels = self.labels

        for i, ax_i in enumerate(axes):
            ax_i.plot(t, U[:, i], label=labels[i])
            ax_i.set_ylabel(f"{labels[i]} {f'[{self.units}]' if self.units else ''}".strip())
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.legend()
            ax_i.grid(True, which="both")

        for j in range(n_ctrl, nrows * ncols):
            r, c = divmod(j, ncols)
            ax_unused = ax.figure.add_subplot(gs[r, c])
            ax_unused.axis("off")

        for ax_i in axes[-ncols:]:
            ax_i.set_xlabel("Time [s]")

        axes[0].set_title(self.title, loc="left", pad=10)
