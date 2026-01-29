from __future__ import annotations

__all__ = ["TargetPlot"]

import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from ADCS.helpers.math_helpers import rot_mat, normalize, quat_diff


def _normalize_modes(modes: list[str] | None) -> list[str]:
    if modes is None or len(modes) == 0:
        return ["real_target"]
    allowed = {"real_target", "est_target", "real_est", "directions3d"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Invalid modes {bad}. Allowed: {sorted(allowed)}")
    out: list[str] = []
    for m in modes:
        if m not in out:
            out.append(m)
    return out


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    if u.size != 3 or v.size != 3:
        return float("nan")
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    u = u / nu
    v = v / nv
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def _boresight_eci(q_b2i: np.ndarray, bore_body_unit: np.ndarray) -> np.ndarray:
    """
    Convert a (Body->ECI) attitude quaternion into the boresight direction in ECI.
    """
    q_b2i = normalize(np.asarray(q_b2i, dtype=float).reshape(4))
    return normalize(rot_mat(q_b2i) @ bore_body_unit)


def _attitude_error_deg(q_b2i: np.ndarray, qref_b2i: np.ndarray) -> float:
    """
    Hamilton, scalar-first.
    Returns the minimal rotation angle between q and q_ref in degrees.
    """
    q_b2i = normalize(np.asarray(q_b2i, dtype=float).reshape(4))
    qref_b2i = normalize(np.asarray(qref_b2i, dtype=float).reshape(4))

    # quat_diff(q0, q1) returns q0^{-1} ⊗ q1 (forced to positive scalar part)
    q_err = quat_diff(q_b2i, qref_b2i)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(w)))


class TargetPlot(Subplot):
    r"""
    Comparison plot between spacecraft boresight / attitude targets and estimates.

    This plot uses sim.target_hist rows of length 4 and supports two encodings:

      1) Vector goal: [nan, tx, ty, tz]
         - Interpreted as a *direction in ECI* (tx,ty,tz).
         - Error (real_target / est_target) is the angle between the spacecraft
           boresight (rotated into ECI) and the target direction.

      2) Quaternion goal: [q0, q1, q2, q3]
         - Interpreted as a *full desired attitude* quaternion (Body->ECI).
         - Error (real_target / est_target) is the *attitude error angle* between
           the current quaternion and q_ref via the relative quaternion.

    Supported modes:
      - real_target: (vector target → boresight error) OR (quat target → attitude error)
      - est_target:  same as above but using est_state_hist
      - real_est:    angle(real boresight, estimated boresight) (requires boresight)
      - directions3d: 3D snapshot of directions at a chosen sample
                      (requires boresight; for quaternion targets, shows boresight
                       implied by q_ref for visualization only)

    Parameters
    ----------
    time : str
        Name of the simulation attribute containing the time vector in seconds.
    title : str
        Title displayed at the top of the plot.
    units : str
        Units used for angular error display, typically degrees.
    modes : list[str] | None
        Which comparisons to display.
    sample_index : int
        Index for the directions3d snapshot (negative selects from end).
    """

    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Target Tracking",
        units: str = "deg",
        modes: list[str] | None = None,
        sample_index: int = -1,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.modes = _normalize_modes(modes)
        self.sample_index = sample_index

    def plot(self, ax, sim) -> None:
        if sim.state_hist is None or getattr(sim, "target_hist", None) is None:
            self._plot_no_data(ax)
            return
        if getattr(sim, "satellite", None) is None:
            self._plot_no_data(ax, msg="No satellite attached to sim_results")
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

        # boresight in body (optional; only needed for vector targets / boresight comparisons)
        bore_body_unit = None
        bore_body = getattr(sim.satellite, "boresight", None)
        if bore_body is not None:
            bb = np.asarray(bore_body, dtype=float).reshape(-1)
            if bb.size == 3 and np.linalg.norm(bb) > 0:
                bore_body_unit = bb / np.linalg.norm(bb)

        # Interpret target_hist rows
        is_quat = np.zeros(N, dtype=bool)
        target_ok = np.ones(N, dtype=bool)

        target_vec_eci = np.full((N, 3), np.nan, dtype=float)  # for vector targets (unit)
        target_qref = np.full((N, 4), np.nan, dtype=float)     # for quaternion targets (unit)

        for i in range(N):
            row = np.asarray(Th[i], dtype=float).reshape(-1)
            if row.size != 4:
                target_ok[i] = False
                continue

            if np.isnan(row[0]):
                # Vector target: [nan, tx, ty, tz]
                v = row[1:4]
                if not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
                    target_ok[i] = False
                    continue
                if bore_body_unit is None:
                    # Can't evaluate boresight error without boresight
                    target_ok[i] = False
                    continue
                target_vec_eci[i] = v / np.linalg.norm(v)
            else:
                # Quaternion target: [q0,q1,q2,q3]
                qref = row
                if not np.all(np.isfinite(qref)) or np.linalg.norm(qref) == 0:
                    target_ok[i] = False
                    continue
                is_quat[i] = True
                target_qref[i] = qref / np.linalg.norm(qref)

        # Build series
        series: dict[str, np.ndarray | None] = {}

        # --- real_target ---
        if "real_target" in self.modes:
            y = np.full(N, np.nan, dtype=float)
            for i in range(N):
                if not target_ok[i]:
                    continue
                q = X_real[i, 3:7]
                if is_quat[i]:
                    y[i] = _attitude_error_deg(q, target_qref[i])
                else:
                    # vector target → boresight error
                    bore_eci = _boresight_eci(q, bore_body_unit)  # type: ignore[arg-type]
                    y[i] = _angle_deg(bore_eci, target_vec_eci[i])
            series["Real vs Target"] = y

        # --- est_target ---
        if "est_target" in self.modes:
            if X_est is None:
                series["Estimated vs Target (missing est_state_hist)"] = None
            else:
                y = np.full(N, np.nan, dtype=float)
                for i in range(N):
                    if not target_ok[i]:
                        continue
                    qh = X_est[i, 3:7]
                    if is_quat[i]:
                        y[i] = _attitude_error_deg(qh, target_qref[i])
                    else:
                        bore_eci_hat = _boresight_eci(qh, bore_body_unit)  # type: ignore[arg-type]
                        y[i] = _angle_deg(bore_eci_hat, target_vec_eci[i])
                series["Estimated vs Target"] = y

        # --- real_est (boresight comparison) ---
        if "real_est" in self.modes:
            if X_est is None:
                series["Real vs Estimated (missing est_state_hist)"] = None
            elif bore_body_unit is None:
                series["Real vs Estimated (missing satellite.boresight)"] = None
            else:
                y = np.full(N, np.nan, dtype=float)
                for i in range(N):
                    q = X_real[i, 3:7]
                    qh = X_est[i, 3:7]
                    bore_eci = _boresight_eci(q, bore_body_unit)
                    bore_eci_hat = _boresight_eci(qh, bore_body_unit)
                    y[i] = _angle_deg(bore_eci, bore_eci_hat)
                series["Real vs Estimated"] = y

        want_3d = "directions3d" in self.modes

        # Layout / availability checks
        n_series = sum(1 for v in series.values() if v is not None)
        if n_series == 0 and not want_3d:
            self._plot_no_data(ax, msg="No valid comparison modes available")
            return
        if want_3d and bore_body_unit is None:
            # directions3d is inherently a boresight-direction visualization
            want_3d = False

        # remove container axis and build sub-axes
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        rows = n_series + (1 if want_3d else 0)
        gs = gridspec.GridSpecFromSubplotSpec(rows, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35)

        r = 0
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

        # Optional 3D snapshot
        if want_3d:
            ax3 = ax.figure.add_subplot(gs[r, 0], projection="3d")

            idx = self.sample_index
            if idx < 0:
                idx = N + idx
            idx = int(np.clip(idx, 0, N - 1))

            q = X_real[idx, 3:7]
            bore_eci = _boresight_eci(q, bore_body_unit)  # type: ignore[arg-type]

            # Target direction for visualization:
            # - vector target: plot the target vector
            # - quaternion target: plot the boresight implied by q_ref (visual aid only)
            target_dir = None
            if target_ok[idx]:
                if is_quat[idx]:
                    qref = target_qref[idx]
                    target_dir = _boresight_eci(qref, bore_body_unit)  # type: ignore[arg-type]
                    target_label = "Target (q_ref boresight)"
                else:
                    target_dir = target_vec_eci[idx]
                    target_label = "Target"
            else:
                target_label = "Target"

            bore_hat = None
            if X_est is not None:
                qh = X_est[idx, 3:7]
                bore_hat = _boresight_eci(qh, bore_body_unit)  # type: ignore[arg-type]

            def _seg(v: np.ndarray):
                return np.array([0.0, v[0]]), np.array([0.0, v[1]]), np.array([0.0, v[2]])

            xs, ys, zs = _seg(bore_eci)
            ax3.plot(xs, ys, zs, label="Real boresight")

            if target_dir is not None and np.all(np.isfinite(target_dir)) and np.linalg.norm(target_dir) > 0:
                xs, ys, zs = _seg(target_dir)
                ax3.plot(xs, ys, zs, label=target_label)

            if bore_hat is not None:
                xs, ys, zs = _seg(bore_hat)
                ax3.plot(xs, ys, zs, label="Estimated boresight")

            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.set_zlim(-1, 1)
            ax3.set_box_aspect([1, 1, 1])
            ax3.set_title(f"Direction snapshot (k={idx})")
            ax3.legend()

    def _plot_no_data(self, ax, msg: str = "No target / state data available") -> None:
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
