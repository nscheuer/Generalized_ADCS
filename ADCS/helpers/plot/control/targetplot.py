__all__ = ["TargetPlot"]

import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from ADCS.helpers.math_helpers import rot_mat


def _normalize_modes(modes):
    if modes is None or len(modes) == 0:
        return ["real_target"]
    allowed = {"real_target", "est_target", "real_est", "directions3d"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Invalid modes {bad}. Allowed: {sorted(allowed)}")
    # de-dup keep order
    out = []
    for m in modes:
        if m not in out:
            out.append(m)
    return out


def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _quat_boresight_eci(q, bore_body):
    """
    Convert a (Body->ECI) attitude quaternion into the boresight direction in ECI.
    """
    R_b2i = rot_mat(q)  # Body -> ECI
    v = R_b2i @ bore_body
    return _safe_unit(v)


def _angle_deg(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    u = u / nu
    v = v / nv
    dot = np.clip(float(np.dot(u, v)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


def _target_direction_eci_from_row(row4: np.ndarray, bore_body: np.ndarray) -> np.ndarray | None:
    """
    Interpret a single target_hist row and return the desired target direction in ECI as a unit vector.

    target_hist row format (len=4):
      - Vector goal: [nan, tx, ty, tz]  -> direction = [tx, ty, tz]
      - Quaternion goal: [q0, q1, q2, q3] -> direction = (rot_mat(q_target) @ bore_body)

    Returns:
      - unit 3-vector (np.ndarray shape (3,))
      - None if the row cannot be interpreted or is invalid
    """
    row4 = np.asarray(row4, dtype=float).reshape(-1)
    if row4.size != 4:
        return None

    # Vector goal if first entry is NaN
    if np.isnan(row4[0]):
        v = row4[1:4]
        if np.linalg.norm(v) == 0:
            return None
        return _safe_unit(v)

    # Quaternion goal otherwise
    q = row4
    if np.linalg.norm(q) == 0:
        return None
    q = q / np.linalg.norm(q)
    v = _quat_boresight_eci(q, bore_body)
    if np.linalg.norm(v) == 0:
        return None
    return v


class TargetPlot(Subplot):
    r"""
    Comparison plot between spacecraft boresight, target direction, and estimates.

    This plot compares boresight direction (in ECI) against a target "goal" that can be:

      1) A vector goal stored in sim.target_hist as [nan, tx, ty, tz]
         where [tx,ty,tz] is the target direction in ECI.

      2) A quaternion goal stored in sim.target_hist as [q0, q1, q2, q3]
         interpreted as a desired attitude quaternion (Body->ECI). In this case,
         the target direction is taken as the desired boresight direction in ECI:
             bore_target_eci = rot_mat(q_target) @ bore_body

    Supported modes:
      - real_target: angle(real boresight, target)
      - est_target: angle(estimated boresight, target)
      - real_est: angle(real boresight, estimated boresight)
      - directions3d: 3D snapshot of directions at a chosen sample

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Units used for angular error display, typically degrees.
    :type units:
        str

    :param modes:
        List of comparison modes to display. Supported values are
        real_target, est_target, real_est, and directions3d.
        If None, the default mode real_target is used.
    :type modes:
        list[str] or None

    :param sample_index:
        Index of the sample used for the three-dimensional direction snapshot
        when directions3d mode is enabled. A negative value selects from the end.
    :type sample_index:
        int
    """

    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Target / Boresight Comparison",
        units: str = "deg",
        modes: list[str] | None = None,
        sample_index: int = -1,  # -1 means last sample
    ):
        self.time = time
        self.title = title
        self.units = units
        self.modes = _normalize_modes(modes)
        self.sample_index = sample_index

    def plot(self, ax, sim) -> None:
        if (
            sim.state_hist is None
            or getattr(sim, "target_hist", None) is None
            or sim.satellite is None
            or not hasattr(sim.satellite, "boresight")
        ):
            self._plot_no_data(ax)
            return

        X_real = np.asarray(sim.state_hist)
        Th = np.asarray(sim.target_hist)

        N = min(len(X_real), len(Th))
        if N <= 0:
            self._plot_no_data(ax)
            return

        X_est = None
        if getattr(sim, "est_state_hist", None) is not None and len(sim.est_state_hist) > 0:
            X_est = np.asarray(sim.est_state_hist)
            N = min(N, len(X_est))

        # time axis
        t = getattr(sim, self.time, None)
        if t is not None:
            t = np.asarray(t)[:N]

        # boresight in body
        bore_body = np.asarray(sim.satellite.boresight, dtype=float).reshape(-1)
        if bore_body.size != 3:
            self._plot_no_data(ax, msg="Satellite boresight must be a 3-vector")
            return
        nb = np.linalg.norm(bore_body)
        if nb == 0:
            self._plot_no_data(ax, msg="Satellite boresight has zero norm")
            return
        bore_body = bore_body / nb

        # Precompute target directions in ECI for all samples (unit vectors)
        target_dirs = np.empty((N, 3), dtype=float)
        target_valid = np.ones(N, dtype=bool)
        for i in range(N):
            v = _target_direction_eci_from_row(Th[i], bore_body)
            if v is None or np.linalg.norm(v) == 0:
                target_dirs[i] = np.array([np.nan, np.nan, np.nan], dtype=float)
                target_valid[i] = False
            else:
                target_dirs[i] = v

        # compute requested series
        series: dict[str, np.ndarray | None] = {}

        if "real_target" in self.modes:
            y = np.full(N, np.nan, dtype=float)
            for i in range(N):
                if not target_valid[i]:
                    continue
                q = X_real[i, 3:7]
                bore_eci = _quat_boresight_eci(q, bore_body)
                y[i] = _angle_deg(bore_eci, target_dirs[i])
            series["Real vs Target"] = y

        if "est_target" in self.modes:
            if X_est is None:
                series["Estimated vs Target (missing est_state_hist)"] = None
            else:
                y = np.full(N, np.nan, dtype=float)
                for i in range(N):
                    if not target_valid[i]:
                        continue
                    qh = X_est[i, 3:7]
                    bore_eci_hat = _quat_boresight_eci(qh, bore_body)
                    y[i] = _angle_deg(bore_eci_hat, target_dirs[i])
                series["Estimated vs Target"] = y

        if "real_est" in self.modes:
            if X_est is None:
                series["Real vs Estimated (missing est_state_hist)"] = None
            else:
                y = np.full(N, np.nan, dtype=float)
                for i in range(N):
                    q = X_real[i, 3:7]
                    qh = X_est[i, 3:7]
                    bore_eci = _quat_boresight_eci(q, bore_body)
                    bore_eci_hat = _quat_boresight_eci(qh, bore_body)
                    y[i] = _angle_deg(bore_eci, bore_eci_hat)
                series["Real vs Estimated"] = y

        want_3d = "directions3d" in self.modes

        # ---- layout ----
        n_series = sum(1 for v in series.values() if v is not None)
        if n_series == 0 and not want_3d:
            self._plot_no_data(ax, msg="No valid comparison modes available")
            return

        # remove the container axis and build sub-axes
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # rows: all time-series + optional 3D directions
        rows = n_series + (1 if want_3d else 0)
        gs = gridspec.GridSpecFromSubplotSpec(
            rows, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35
        )

        r = 0
        # time-series axes
        for name, y in series.items():
            if y is None:
                continue
            ax_i = ax.figure.add_subplot(gs[r, 0])
            if t is not None:
                ax_i.plot(t, y, label=name)
                ax_i.set_xlabel("Time [s]")
            else:
                ax_i.plot(y, label=name)
                ax_i.set_xlabel("Sample")
            ax_i.set_ylabel(f"Error [{self.units}]")
            ax_i.grid(True, which="both")
            ax_i.legend()
            if r == 0:
                ax_i.set_title(self.title, loc="left", pad=10)
            r += 1

        # optional 3D “direction snapshot”
        if want_3d:
            ax3 = ax.figure.add_subplot(gs[r, 0], projection="3d")

            idx = self.sample_index
            if idx < 0:
                idx = N + idx  # python-style from end
            idx = int(np.clip(idx, 0, N - 1))

            q = X_real[idx, 3:7]
            bore_eci = _quat_boresight_eci(q, bore_body)

            target = target_dirs[idx]
            if not np.all(np.isfinite(target)) or np.linalg.norm(target) == 0:
                target = np.array([np.nan, np.nan, np.nan], dtype=float)

            bore_hat = None
            if X_est is not None:
                qh = X_est[idx, 3:7]
                bore_hat = _quat_boresight_eci(qh, bore_body)

            # draw unit vectors from origin
            def _vec(v):
                return np.array([0, v[0]]), np.array([0, v[1]]), np.array([0, v[2]])

            xs, ys, zs = _vec(bore_eci)
            ax3.plot(xs, ys, zs, label="Real boresight")

            if np.all(np.isfinite(target)):
                xs, ys, zs = _vec(target)
                ax3.plot(xs, ys, zs, label="Target")
            else:
                # still show legend entry if desired; keep it simple and skip plotting invalid target
                pass

            if bore_hat is not None:
                xs, ys, zs = _vec(bore_hat)
                ax3.plot(xs, ys, zs, label="Estimated boresight")

            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.set_zlim(-1, 1)
            ax3.set_box_aspect([1, 1, 1])
            ax3.set_title(f"Direction snapshot (k={idx})")
            ax3.legend()

    def _plot_no_data(self, ax, msg="No target / state data available"):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
