import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


class OrbitVelocityPlot(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Orbit Velocity (ECI)",
        units: str = "km/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        mag_color: str = "tab:red",
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.mag_color = mag_color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

        ax_vx = ax.figure.add_subplot(gs[0, 0])
        ax_vmag = ax.figure.add_subplot(gs[0, 1])
        ax_vy = ax.figure.add_subplot(gs[1, 0])
        ax_vz = ax.figure.add_subplot(gs[1, 1])

        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        V = np.vstack([np.asarray(os.V) for os in sim.os_hist])
        vmag = np.linalg.norm(V, axis=1)

        labels = [r"$v_x$", r"$v_y$", r"$v_z$"]
        axes = [ax_vx, ax_vy, ax_vz]

        for i, ax_i in enumerate(axes):
            ax_i.plot(t, V[:, i], color=self.colors[i], label=labels[i])
            ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.legend()
            ax_i.grid(True, which="both")

        ax_vmag.plot(t, vmag, color=self.mag_color, label=r"$\|v\|$")
        ax_vmag.set_ylabel(f"$\\|v\\|$ [{self.units}]")
        if self.log_y:
            ax_vmag.set_yscale("log")
        ax_vmag.legend()
        ax_vmag.grid(True, which="both")

        ax_vy.set_xlabel("Time [s]")
        ax_vz.set_xlabel("Time [s]")
        ax_vx.set_title(self.title, loc="left", pad=10)


class OrbitVelocityPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: str,
        time: str = "time_s",
        title: str | None = None,
        units: str = "km/s",
        color: str | None = None,
        log_y: bool = False,
        labels: dict[str, str] | None = None,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")
        self.component = component
        self.time = time
        self.units = units
        self.color = color
        self.log_y = log_y
        self.labels = labels or {
            "x": r"$v_x$",
            "y": r"$v_y$",
            "z": r"$v_z$",
            "m": r"$\|v\|$",
        }
        self.title = title

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title or "Orbit Velocity (ECI)", loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        V = np.vstack([np.asarray(os.V) for os in sim.os_hist])

        if self.component == "x":
            y = V[:, 0]
            label = self.labels["x"]
            color = self.color or "tab:blue"
            title = self.title or "Orbit Velocity $v_x$ (ECI)"
        elif self.component == "y":
            y = V[:, 1]
            label = self.labels["y"]
            color = self.color or "tab:orange"
            title = self.title or "Orbit Velocity $v_y$ (ECI)"
        elif self.component == "z":
            y = V[:, 2]
            label = self.labels["z"]
            color = self.color or "tab:green"
            title = self.title or "Orbit Velocity $v_z$ (ECI)"
        else:
            y = np.linalg.norm(V, axis=1)
            label = self.labels["m"]
            color = self.color or "tab:red"
            title = self.title or "Orbit Velocity $\\|v\\|$ (ECI)"

        ax.plot(t, y, color=color, label=label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [{self.units}]")
        ax.set_title(title)
        if self.log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both")


class OrbitVelocityPlotCombined(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Orbit Velocity (ECI)",
        units: str = "km/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
        labels: list[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y
        self.labels = labels or [r"$v_x$", r"$v_y$", r"$v_z$"]

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        V = np.vstack([np.asarray(os.V) for os in sim.os_hist])

        for i in range(3):
            ax.plot(
                t,
                V[:, i],
                color=self.colors[i],
                label=self.labels[i],
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Velocity [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, which="both")