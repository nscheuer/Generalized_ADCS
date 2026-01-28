__all__ = ["AttitudePlot"]

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons

from ..subplot import Subplot
from ADCS.helpers.math_helpers import rot_mat


def _normalize_attitude_sources(sources: Optional[list[str]]) -> list[str]:
    if not sources:
        return ["real"]  # default
    allowed = {"real", "estimated", "reference"}
    out: list[str] = []
    for s in sources:
        s2 = str(s).strip().lower()
        if s2 not in allowed:
            raise ValueError(f"Invalid source {s!r}. Allowed: {sorted(allowed)}")
        if s2 not in out:
            out.append(s2)
    return out


def _get_time(sim, time_attr: str) -> Optional[np.ndarray]:
    t = getattr(sim, time_attr, None)
    if t is None:
        return None
    t = np.asarray(t)
    return t if t.size > 0 else None


class AttitudePlot(Subplot):
    r"""
    Interactive 3D attitude animation subplot wrapper.

    This class represents a specialized :class:`~ADCS.subplot.Subplot` that launches
    an external Matplotlib 3D animation window to visualize spacecraft attitude
    evolution in an inertial reference frame. The animation depicts body-fixed axes,
    optional estimated axes, reference directions, and environmental vectors such as
    magnetic field and Sun direction.

    The spacecraft attitude is assumed to be represented by unit quaternions
    :math:`\mathbf{q} = [q_x, q_y, q_z, q_w]^\top`, stored in the simulation history.
    For each quaternion, a direction cosine matrix is computed as

    .. math::

        \mathbf{R}(\mathbf{q}) =
        \begin{bmatrix}
        1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z q_w) & 2(q_x q_z + q_y q_w) \\
        2(q_x q_y + q_z q_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x q_w) \\
        2(q_x q_z - q_y q_w) & 2(q_y q_z + q_x q_w) & 1 - 2(q_x^2 + q_y^2)
        \end{bmatrix}

    which maps body-frame vectors into the inertial frame. The inertial components
    of the body axes are then given by

    .. math::

        \mathbf{a}_i^{\text{ECI}} = \mathbf{R}(\mathbf{q}) \, \mathbf{e}_i

    where :math:`\mathbf{e}_i` denotes the canonical unit vectors of the body frame.

    A reference vector :math:`\mathbf{g}(t) \in \mathbb{R}^3`, such as a target
    direction expressed in the inertial frame, may be displayed after normalization

    .. math::

        \hat{\mathbf{g}}(t) = \frac{\mathbf{g}(t)}{\lVert \mathbf{g}(t) \rVert}

    Environmental vectors, when enabled, are visualized in a similar normalized form.
    Animation playback supports pause, resume, and variable playback speed.

    The subplot axis itself is not used for drawing; instead, it displays a textual
    notice while the animation is shown in a separate window.

    :param sources:
        List of attitude or vector sources to display. Valid entries are ``"real"``,
        ``"estimated"``, and ``"reference"``.
    :type sources:
        list[str] or None

    :param time:
        Name of the simulation attribute containing the time history in seconds.
    :type time:
        str

    :param title:
        Title of the animation window.
    :type title:
        str

    :param reference_attr:
        Name of the simulation attribute containing the reference vector history
        expressed in the inertial frame.
    :type reference_attr:
        str

    :param body_axis:
        Body-frame axis to compare against the reference direction. Must be ``"x"``,
        ``"y"``, or ``"z"``.
    :type body_axis:
        str

    :param axis_limits:
        Symmetric plot limits for each inertial axis.
    :type axis_limits:
        float

    :param interval_ms:
        Animation update interval in milliseconds.
    :type interval_ms:
        int

    :param show_env:
        Flag indicating whether environmental vectors from the simulation state
        should be displayed.
    :type show_env:
        bool

    """

    def __init__(
        self,
        *,
        sources: Optional[list[str]] = None,  # ["real","estimated","reference"]
        time: str = "time_s",
        title: str = "Attitude Animation (ECI)",
        reference_attr: str = "eci_target_hist",  # Nx3 ECI direction (or None)
        body_axis: str = "z",  # which body axis to compare to reference: "x","y","z"
        axis_limits: float = 1.0,
        interval_ms: int = 50,
        show_env: bool = True,  # uses sim.os_hist B/S if available
    ):
        self.sources = _normalize_attitude_sources(sources)
        self.time = time
        self.title = title
        self.reference_attr = reference_attr
        self.body_axis = body_axis.lower()
        if self.body_axis not in {"x", "y", "z"}:
            raise ValueError("body_axis must be one of: 'x', 'y', 'z'")
        self.axis_limits = float(axis_limits)
        self.interval_ms = int(interval_ms)
        self.show_env = bool(show_env)

    def plot(self, ax, sim) -> None:
        # Keep the subplot clean and indicate that a separate window is created.
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "Launching 3D attitude animation…\n(close it to continue)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        self._run_animation(sim)

    # ---------------- Internals ----------------

    def _get_q_series(self, sim, which: str) -> Optional[np.ndarray]:
        if which == "real":
            X = getattr(sim, "state_hist", None)
        elif which == "estimated":
            X = getattr(sim, "est_state_hist", None)
        else:
            raise ValueError(which)

        if X is None or len(X) == 0:
            return None

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] < 7:
            return None

        return np.asarray(X[:, 3:7], dtype=float)

    def _get_reference_series(self, sim) -> Optional[np.ndarray]:
        if self.reference_attr is None:
            return None
        G = getattr(sim, self.reference_attr, None)
        if G is None or len(G) == 0:
            return None
        G = np.asarray(G, dtype=float)
        if G.ndim != 2 or G.shape[1] != 3:
            return None
        return G

    def _run_animation(self, sim) -> None:
        t = _get_time(sim, self.time)

        q_real = self._get_q_series(sim, "real") if "real" in self.sources else None
        q_est = self._get_q_series(sim, "estimated") if "estimated" in self.sources else None
        ref = self._get_reference_series(sim) if "reference" in self.sources else None

        os_hist = getattr(sim, "os_hist", None) if self.show_env else None

        # Determine N (safe)
        lengths = []
        if q_real is not None:
            lengths.append(q_real.shape[0])
        if q_est is not None:
            lengths.append(q_est.shape[0])
        if ref is not None:
            lengths.append(ref.shape[0])
        if os_hist is not None:
            lengths.append(len(os_hist))

        if not lengths:
            # nothing to animate
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(self.title, loc="left")
            ax.text(0.5, 0.5, "No attitude/reference data available", ha="center", va="center")
            plt.show()
            return

        N = int(min(lengths))
        if N <= 0:
            return

        # Time fallback
        if t is None or np.asarray(t).size < N:
            t_use = np.arange(N, dtype=float)
            xlab = "Sample"
        else:
            t_use = np.asarray(t, dtype=float)[:N]
            xlab = "Time [s]"

        # 3D figure
        fig = plt.figure(figsize=(9, 9))
        ax3 = fig.add_subplot(111, projection="3d")

        L = self.axis_limits
        ax3.set_xlim([-L, L])
        ax3.set_ylim([-L, L])
        ax3.set_zlim([-L, L])
        ax3.set_xlabel("X (ECI)")
        ax3.set_ylabel("Y (ECI)")
        ax3.set_zlabel("Z (ECI)")

        title_parts = []
        if q_real is not None:
            title_parts.append("True Att")
        if q_est is not None:
            title_parts.append("Est Att")
        if ref is not None:
            title_parts.append("Ref")
        if os_hist is not None:
            title_parts.append("Env")

        ax3.set_title(self.title + ("  |  " + " + ".join(title_parts) if title_parts else ""))

        body_axes = np.eye(3, dtype=float)
        axis_index = {"x": 0, "y": 1, "z": 2}[self.body_axis]

        # --- Artists ---
        true_lines = []
        if q_real is not None:
            colors = ["r", "g", "b"]
            true_lines = [
                ax3.plot([], [], [], lw=2, color=colors[k], label=f"True {axname}")[0]
                for k, axname in enumerate(["X", "Y", "Z"])
            ]

        est_lines = []
        if q_est is not None:
            colors = ["salmon", "lightgreen", "lightblue"]
            est_lines = [ax3.plot([], [], [], lw=1, linestyle="--", color=colors[k])[0] for k in range(3)]

        ref_line = None
        if ref is not None:
            ref_line = ax3.plot([], [], [], lw=2, linestyle=":", color="cyan", label="Reference")[0]

        # Environment quivers (recreated each frame, like your original)
        B_arrow = None
        S_arrow = None

        # Legend proxies (keeps legend short & stable)
        proxies, labels = [], []
        if q_real is not None:
            proxies.append(plt.Line2D([0], [0], color="r", lw=2))
            labels.append("True body axes")
        if q_est is not None:
            proxies.append(plt.Line2D([0], [0], color="salmon", lw=1, linestyle="--"))
            labels.append("Estimated body axes")
        if ref is not None:
            proxies.append(plt.Line2D([0], [0], color="cyan", lw=2, linestyle=":"))
            labels.append("Reference vector")
        if os_hist is not None:
            proxies.append(plt.Line2D([0], [0], color="magenta", lw=2))
            labels.append("B-field")
            proxies.append(plt.Line2D([0], [0], color="orange", lw=2))
            labels.append("Sun")

        if proxies:
            ax3.legend(proxies, labels, loc="upper left")

        # --- Controls ---
        frame = [0.0]
        play = [True]
        speed = [1.0]

        def init_anim():
            artists = []
            artists.extend(true_lines)
            artists.extend(est_lines)
            if ref_line is not None:
                artists.append(ref_line)
            return artists

        def update(_):
            nonlocal B_arrow, S_arrow

            if not play[0]:
                return []

            frame[0] = (frame[0] + speed[0]) % N
            i = int(frame[0])

            # True
            if q_real is not None:
                Rt = rot_mat(q_real[i])  # body -> ECI
                true_ax = Rt @ body_axes
                for k in range(3):
                    true_lines[k].set_data([0, true_ax[0, k]], [0, true_ax[1, k]])
                    true_lines[k].set_3d_properties([0, true_ax[2, k]])

            # Estimated
            if q_est is not None:
                Re = rot_mat(q_est[i])
                est_ax = Re @ body_axes
                for k in range(3):
                    est_lines[k].set_data([0, est_ax[0, k]], [0, est_ax[1, k]])
                    est_lines[k].set_3d_properties([0, est_ax[2, k]])

            # Reference vector
            if ref_line is not None:
                g = ref[i].astype(float)
                ng = np.linalg.norm(g)
                if ng > 1e-9:
                    g = g / ng
                    ref_line.set_data([0, g[0]], [0, g[1]])
                    ref_line.set_3d_properties([0, g[2]])
                else:
                    ref_line.set_data([], [])
                    ref_line.set_3d_properties([])

            # Environment (recreate quivers each frame)
            if os_hist is not None and i < len(os_hist) and os_hist[i] is not None:
                if B_arrow is not None:
                    B_arrow.remove()
                if S_arrow is not None:
                    S_arrow.remove()

                os_i = os_hist[i]

                # B field
                B = getattr(os_i, "B", None)
                if B is not None:
                    B = np.asarray(B, dtype=float).reshape(3)
                    nB = np.linalg.norm(B)
                    if nB > 1e-9:
                        b = B / nB
                        B_arrow = ax3.quiver(0, 0, 0, b[0], b[1], b[2], color="magenta")
                    else:
                        B_arrow = ax3.quiver(0, 0, 0, 0, 0, 0, color="magenta", alpha=0)

                # Sun
                S = getattr(os_i, "S", None)
                if S is not None:
                    is_lit = True
                    if hasattr(os_i, "is_sunlit"):
                        try:
                            is_lit = bool(os_i.is_sunlit())
                        except Exception:
                            is_lit = True

                    if is_lit:
                        S = np.asarray(S, dtype=float).reshape(3)
                        nS = np.linalg.norm(S)
                        if nS > 1e-9:
                            s = S / nS
                            S_arrow = ax3.quiver(0, 0, 0, s[0], s[1], s[2], color="orange")
                        else:
                            S_arrow = ax3.quiver(0, 0, 0, 0, 0, 0, color="orange", alpha=0)
                    else:
                        S_arrow = ax3.quiver(0, 0, 0, 0, 0, 0, color="grey", alpha=0)

            # Show a tiny time readout in window title (optional, cheap)
            try:
                fig.canvas.manager.set_window_title(f"{self.title} — {xlab}: {t_use[i]:.3f}")
            except Exception:
                pass

            artists = []
            artists.extend(true_lines)
            artists.extend(est_lines)
            if ref_line is not None:
                artists.append(ref_line)
            if B_arrow is not None:
                artists.append(B_arrow)
            if S_arrow is not None:
                artists.append(S_arrow)
            return artists

        ani = FuncAnimation(fig, update, init_func=init_anim, interval=self.interval_ms)

        # UI buttons
        ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
        btn_pause = Button(ax_pause, "Pause / Play")
        btn_pause.on_clicked(lambda _e: play.__setitem__(0, not play[0]))

        ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
        speed_buttons = RadioButtons(ax_speed, ["0.25x", "0.5x", "1x", "2x", "4x"], active=2)

        def set_speed(label: str):
            speed[0] = float(label.replace("x", ""))

        speed_buttons.on_clicked(set_speed)

        fig.animation = ani
