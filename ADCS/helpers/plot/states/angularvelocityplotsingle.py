import numpy as np
from ..subplot import Subplot

class AngularVelocityPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: str,          # 'x', 'y', 'z', or 'm'
        time: str = "time_s",
        title: str | None = None,
        units: str = "rad/s",
        color: str | None = None,
        log_y: bool = False,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")

        self.component = component
        self.time = time
        self.units = units
        self.log_y = log_y

        self.color = color
        self.title = title

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        X = np.vstack(sim.state_hist)
        w = X[:, 0:3]

        if self.component == "x":
            y = w[:, 0]
            label = r"$\omega_x$"
            color = self.color or "tab:blue"
            title = self.title or r"$\omega_x$ vs Time"

        elif self.component == "y":
            y = w[:, 1]
            label = r"$\omega_y$"
            color = self.color or "tab:orange"
            title = self.title or r"$\omega_y$ vs Time"

        elif self.component == "z":
            y = w[:, 2]
            label = r"$\omega_z$"
            color = self.color or "tab:green"
            title = self.title or r"$\omega_z$ vs Time"

        else:  # 'm'
            y = np.linalg.norm(w, axis=1)
            label = r"$\|\omega\|$"
            color = self.color or "tab:red"
            title = self.title or r"$\|\omega\|$ vs Time"

        ax.plot(t, y, color=color, label=label)
        ax.set_ylabel(f"{label} [{self.units}]")
        ax.set_xlabel("Time [s]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both")

        if self.log_y:
            ax.set_yscale("log")
