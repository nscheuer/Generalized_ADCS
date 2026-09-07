"""Quantitative metrics for Monte Carlo / single-run ADCS results.

This module is the single source of the numeric quantities that the
*Generalized Attitude Control Allocation* paper reports in its tables and
``[TODO-DATA]`` slots. It is intentionally dependency-light (numpy only),
pure, and deterministic so it can be reused by data-generation scripts and
unit-tested in isolation.

Result format
-------------
The Monte Carlo workers in ``papers/Generalized_ACS/`` return a plain dict
per run::

    {"run_id", "config", "time": (N,), "state": (N, 7+nw),
     "u": (N, n_act), "boresight_goal": (N, 3) or (N, 4)}

``state[:, 3:7]`` is the scalar-first attitude quaternion ``(w, x, y, z)``.
The pointing-error definition here is byte-for-byte the same one used by
``ADCS.helpers.plotting_mc.plot_controller_mc`` (a unit test cross-checks
the two rotation helpers), so tables and figures never disagree.

Paper artifact map
------------------
* :func:`convergence_stats`         -> TAB-MC, FIG-MC
* :func:`settling_time`,
  :func:`steady_state_error_deg`    -> TAB-SAMELAW, TAB-DIFFLAW
* :func:`allocation_direction_error_deg`,
  :func:`allocation_magnitude_ratio`-> FIG-ALLOC, TAB-ALLOC
* :func:`write_table`               -> emits the .tex/.csv/.md the paper
                                       has no build pipeline for
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "quat_to_R_b2i",
    "boresight_error_deg",
    "run_pointing_error",
    "final_error_deg",
    "settling_time",
    "steady_state_error_deg",
    "converged",
    "convergence_stats",
    "allocation_direction_error_deg",
    "allocation_magnitude_ratio",
    "metrics_table",
    "write_table",
    "run_results_to_dict",
    "from_simulation_results",
    "rw_momentum",
]


# --------------------------------------------------------------------------- #
# Attitude / pointing error
# --------------------------------------------------------------------------- #
def quat_to_R_b2i(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternions ``(w, x, y, z)`` -> body->inertial rotations.

    Input ``q`` is ``(N, 4)``; output is ``(N, 3, 3)``.

    NOTE: this must stay numerically identical to
    ``ADCS.helpers.plotting_mc.plot_controller_mc._rot_mat_vec`` so that the
    paper's tables and figures use one definition. ``test_metrics`` asserts
    this on random quaternions.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def _goal_eci(boresight_goal: np.ndarray) -> np.ndarray:
    """Normalize the stored goal to an ``(N, 3)`` ECI direction.

    Mirrors the shape handling in ``plot_target_tracking_mc``: a width-4
    array is interpreted as ``[?, gx, gy, gz]`` (columns ``1:4`` used).
    """
    g = np.asarray(boresight_goal, dtype=float)
    if g.ndim != 2:
        raise ValueError(f"boresight_goal must be 2-D, got shape {g.shape}")
    if g.shape[1] == 4:
        return g[:, 1:4]
    if g.shape[1] == 3:
        return g
    raise ValueError(f"Unexpected boresight_goal shape: {g.shape}")


def boresight_error_deg(
    state: np.ndarray,
    boresight_goal: np.ndarray,
    body_boresight: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Per-timestep angle (deg) between the body boresight (rotated to ECI)
    and the desired ECI direction.

    Identical definition to ``plot_target_tracking_mc``.
    """
    state = np.asarray(state, dtype=float)
    v_b_body = np.asarray(body_boresight, dtype=float)
    nb = np.linalg.norm(v_b_body)
    if nb == 0:
        raise ValueError("body_boresight must be non-zero.")
    v_b_body = v_b_body / nb

    q_hist = state[:, 3:7]
    R_b2i = quat_to_R_b2i(q_hist)
    v_bore_eci = np.einsum("nij,j->ni", R_b2i, v_b_body)
    goal = _goal_eci(boresight_goal)

    v_b = v_bore_eci / np.linalg.norm(v_bore_eci, axis=1, keepdims=True)
    v_g = goal / np.linalg.norm(goal, axis=1, keepdims=True)
    dot = np.clip(np.sum(v_b * v_g, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def run_pointing_error(
    res: Dict[str, Any],
    body_boresight: Sequence[float] = (0.0, 0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """``(time, error_deg)`` for one MC-result dict."""
    return (
        np.asarray(res["time"], dtype=float),
        boresight_error_deg(res["state"], res["boresight_goal"], body_boresight),
    )


def final_error_deg(
    res: Dict[str, Any], body_boresight: Sequence[float] = (0.0, 0.0, 1.0)
) -> float:
    """Final-timestep pointing error (deg) — the TAB-MC convergence quantity."""
    return float(run_pointing_error(res, body_boresight)[1][-1])


# --------------------------------------------------------------------------- #
# Time-domain performance (TAB-SAMELAW / TAB-DIFFLAW)
# --------------------------------------------------------------------------- #
def settling_time(
    time: np.ndarray,
    error_deg: np.ndarray,
    threshold_deg: float = 5.0,
) -> float:
    """Earliest time after which ``error_deg`` stays at/below ``threshold_deg``
    for the rest of the run.

    Returns the sample time of the first index ``i`` such that
    ``all(error_deg[i:] <= threshold_deg)``. ``time[0]`` if already settled,
    ``nan`` if the final sample is still above threshold (never settles).
    """
    time = np.asarray(time, dtype=float)
    err = np.asarray(error_deg, dtype=float)
    above = err > threshold_deg
    if not above.any():
        return float(time[0])
    if above[-1]:
        return float("nan")
    last_above = int(np.flatnonzero(above)[-1])
    return float(time[last_above + 1])


def steady_state_error_deg(
    time: np.ndarray,
    error_deg: np.ndarray,
    last_frac: float = 0.1,
) -> float:
    """Mean pointing error over the final ``last_frac`` of samples."""
    err = np.asarray(error_deg, dtype=float)
    n = err.shape[0]
    if not 0.0 < last_frac <= 1.0:
        raise ValueError("last_frac must be in (0, 1].")
    k = max(1, int(round(n * last_frac)))
    return float(np.mean(err[-k:]))


def converged(
    res: Dict[str, Any],
    threshold_deg: float = 5.0,
    body_boresight: Sequence[float] = (0.0, 0.0, 1.0),
) -> bool:
    """True iff the final-timestep pointing error is below ``threshold_deg``."""
    return final_error_deg(res, body_boresight) < threshold_deg


def convergence_stats(
    full_results: List[Dict[str, Any]],
    threshold_deg: float = 5.0,
    settle_threshold_deg: Optional[float] = None,
    body_boresight: Sequence[float] = (0.0, 0.0, 1.0),
) -> Dict[str, float]:
    """Aggregate convergence statistics over a Monte Carlo set.

    Produces exactly the quantities Paper 1 reports for TAB-MC / FIG-MC, e.g.
    *"84% converging (<5 deg, mean 3.8 deg)"*:

    ``pct_converged`` (final error < ``threshold_deg``), ``mean_final``,
    ``median_final``, ``min_final``, ``max_final``, ``std_final``,
    ``mean_settle`` (mean settling time over converged runs, using
    ``settle_threshold_deg`` which defaults to ``threshold_deg``), and ``n``.
    """
    if not full_results:
        raise ValueError("full_results is empty.")
    st = threshold_deg if settle_threshold_deg is None else settle_threshold_deg
    finals: List[float] = []
    settles: List[float] = []
    for res in full_results:
        t, err = run_pointing_error(res, body_boresight)
        finals.append(float(err[-1]))
        ts = settling_time(t, err, st)
        if np.isfinite(ts):
            settles.append(ts)
    finals_a = np.asarray(finals, dtype=float)
    converged_mask = finals_a < threshold_deg
    return {
        "n": int(finals_a.size),
        "pct_converged": float(100.0 * np.mean(converged_mask)),
        "n_converged": int(np.count_nonzero(converged_mask)),
        "mean_final": float(np.mean(finals_a)),
        "median_final": float(np.median(finals_a)),
        "min_final": float(np.min(finals_a)),
        "max_final": float(np.max(finals_a)),
        "std_final": float(np.std(finals_a)),
        "mean_settle": float(np.mean(settles)) if settles else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Allocation quality (FIG-ALLOC / TAB-ALLOC) — pure helpers
# --------------------------------------------------------------------------- #
def _as_2d3(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    if v.shape[-1] != 3:
        raise ValueError(f"expected (...,3) torque array, got {v.shape}")
    return v


def allocation_direction_error_deg(
    tau_achieved: np.ndarray,
    tau_desired: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Angle (deg) between achieved and desired torque, per sample.

    This is the Paper 1 §V-E quantity: LP keeps it ~0 (mean 0.004 deg),
    QP can exceed 30 deg. Samples where either vector is ~0 yield ``nan``.
    """
    a = _as_2d3(tau_achieved)
    d = _as_2d3(tau_desired)
    na = np.linalg.norm(a, axis=-1)
    nd = np.linalg.norm(d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(a * d, axis=-1) / (na * nd)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.rad2deg(np.arccos(cos))
    ang[(na < eps) | (nd < eps)] = np.nan
    return ang


def allocation_magnitude_ratio(
    tau_achieved: np.ndarray,
    tau_desired: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Fraction of desired torque magnitude delivered *along the desired
    direction*: ``(tau_achieved . d_hat) / |tau_desired|`` per sample.

    Using the projection (not raw norm ratio) keeps this consistent with the
    LP objective, which maximizes torque along the commanded direction.
    """
    a = _as_2d3(tau_achieved)
    d = _as_2d3(tau_desired)
    nd = np.linalg.norm(d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.sum(a * d, axis=-1) / (nd**2)
    ratio = np.where(nd < eps, np.nan, ratio)
    return ratio


# --------------------------------------------------------------------------- #
# Table emission (paper has no LaTeX build pipeline)
# --------------------------------------------------------------------------- #
def metrics_table(
    rows: List[Dict[str, Any]],
    columns: Optional[Sequence[str]] = None,
    fmt: str = "latex",
    float_fmt: str = "{:.3f}",
) -> str:
    """Render ``rows`` (list of flat dicts) as a ``latex``, ``csv`` or ``md``
    table string. Column order = ``columns`` or first row's key order."""
    if not rows:
        raise ValueError("rows is empty.")
    cols = list(columns) if columns is not None else list(rows[0].keys())

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "nan" if np.isnan(v) else float_fmt.format(v)
        return str(v)

    if fmt == "csv":
        out = [",".join(cols)]
        out += [",".join(cell(r.get(c, "")) for c in cols) for r in rows]
        return "\n".join(out) + "\n"
    if fmt == "md":
        out = ["| " + " | ".join(cols) + " |",
               "| " + " | ".join("---" for _ in cols) + " |"]
        out += ["| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |"
                for r in rows]
        return "\n".join(out) + "\n"
    if fmt == "latex":
        esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
        out = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
               " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
        out += [" & ".join(esc(cell(r.get(c, ""))) for c in cols) + r" \\"
                for r in rows]
        out += [r"\hline", r"\end{tabular}"]
        return "\n".join(out) + "\n"
    raise ValueError(f"unknown fmt {fmt!r} (latex|csv|md)")


def write_table(
    rows: List[Dict[str, Any]],
    path_stem: str,
    columns: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("latex", "csv", "md"),
    float_fmt: str = "{:.3f}",
) -> List[str]:
    """Write ``rows`` next to the generated data as ``<stem>.tex/.csv/.md``.
    Returns the paths written."""
    from pathlib import Path

    ext = {"latex": ".tex", "csv": ".csv", "md": ".md"}
    written: List[str] = []
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        p = stem.with_suffix(ext[f])
        p.write_text(metrics_table(rows, columns=columns, fmt=f,
                                   float_fmt=float_fmt))
        written.append(str(p))
    return written


# --------------------------------------------------------------------------- #
# Structured-result adapters (Paper 2: ADCS.simulate_mc / SimulationResults)
# --------------------------------------------------------------------------- #
# Paper 2's planner scripts use ``ADCS.simulate_mc`` which returns
# ``SimulationResults`` (a list of ``RunResults`` dataclasses) rather than the
# raw worker dicts Paper 1 uses. ``RunResults.state_hist`` and
# ``.target_hist`` (the ``goal.to_ref`` reference, same quantity Paper 1
# stored as ``boresight_goal``) map straight onto the verified core above, so
# every metric works on Paper 2 data with zero new pointing math.
def run_results_to_dict(run: Any) -> Dict[str, Any]:
    """Adapt one ``RunResults`` to the dict shape the metrics consume."""
    d: Dict[str, Any] = {
        "time": np.asarray(run.time_s, dtype=float),
        "state": np.asarray(run.state_hist, dtype=float),
        "boresight_goal": np.asarray(run.target_hist, dtype=float),
    }
    if getattr(run, "control_hist", None) is not None:
        d["u"] = np.asarray(run.control_hist, dtype=float)
    return d


def from_simulation_results(sim_results: Any) -> List[Dict[str, Any]]:
    """``SimulationResults`` (or any iterable of ``RunResults``) ->
    ``list`` of metric-ready dicts. Use the result with
    :func:`convergence_stats`, :func:`run_pointing_error`, etc."""
    runs = getattr(sim_results, "runs", sim_results)
    return [run_results_to_dict(r) for r in runs]


def rw_momentum(res: Dict[str, Any], n_rw: Optional[int] = None) -> np.ndarray:
    """Reaction-wheel stored momentum history ``(N, n_rw)`` from a result's
    state (``state = [w(3), q(4), h(n_rw)]``). For Paper 2 IV-A, which shows
    pointing error and wheel momentum converging simultaneously (implicit
    desaturation). Returns shape ``(N, 0)`` for wheel-less configs."""
    s = np.asarray(res["state"], dtype=float)
    h = s[:, 7:]
    if n_rw is not None:
        h = h[:, :n_rw]
    return h
