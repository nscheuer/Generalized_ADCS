import matplotlib.pyplot as plt
import numpy as np

from ..subplot import Subplot

class AngularVelocityPlotCombined(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Angular Rates (Body Frame)",
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
        t = getattr(sim, self.time)

        X = np.vstack(sim.state_hist)
        w = X[:, 0:3]

        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

        for i in range(3):
            ax.plot(t, w[:, i], color=self.colors[i], label=labels[i])

        ax.set_ylabel(f"Angular Velocity [{self.units}]")
        ax.set_title(self.title)
        
        if self.log_y:
            ax.set_yscale("log")
            
        ax.legend(loc="upper right")
        ax.grid(True, which="both", linestyle='--', alpha=0.7)