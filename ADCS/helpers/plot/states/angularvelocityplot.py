import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..subplot import Subplot

class AngularVelocityPlot(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Angular Rates in Body Frame",
        units: str = "rad/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)

        # Create a gridspec *inside* the provided axis
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=ax.get_subplotspec()
        )

        sub_axes_wx  = ax.figure.add_subplot(gs[0, 0])
        sub_axes_mag = ax.figure.add_subplot(gs[0, 1])
        sub_axes_wy  = ax.figure.add_subplot(gs[1, 0])
        sub_axes_wz  = ax.figure.add_subplot(gs[1, 1])

        t = getattr(sim, self.time)

        X = np.vstack(sim.state_hist)
        w = X[:, 0:3]
        w_mag = np.linalg.norm(w, axis=1)

        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
        axes = [sub_axes_wx, sub_axes_wy, sub_axes_wz]

        for i, ax_i in enumerate(axes):
            ax_i.plot(t, w[:, i], color=self.colors[i], label=labels[i])
            ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.legend()
            ax_i.grid(True, which="both")

        sub_axes_mag.plot(t, w_mag, color="tab:red", label=r"$\|\omega\|$")
        sub_axes_mag.set_ylabel(f"$\|\omega\|$ [{self.units}]")
        if self.log_y:
            sub_axes_mag.set_yscale("log")
        sub_axes_mag.legend()
        sub_axes_mag.grid(True, which="both")

        sub_axes_wy.set_xlabel("Time [s]")
        sub_axes_wz.set_xlabel("Time [s]")

        sub_axes_wx.set_title(self.title, loc="left", pad=10)