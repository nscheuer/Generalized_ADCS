"""Trajectory quality metrics for Monte Carlo analysis.

Provides functions to compute quality scores from simulated trajectory data
(MC result dicts). The composite score is consistent with the plan-time metric
used in ``Plan_and_Track_LQR._plan_quality_score``.

Quality score = settle_frac(5°) + tail_mean / 180°

- settle_frac: fraction of trajectory before error permanently drops below 5°.
  0 = converged instantly, 1 = never settled.
- tail_mean / 180°: mean error over last 50% of trajectory, normalized.
  0 = perfect, 1 = worst.

Score range [0, 2].  ★★★ < 0.3,  ★★ < 0.5,  ★ < 1.0,  ✗ ≥ 1.0
"""

from __future__ import annotations

__all__ = [
    "compute_error_trace",
    "settle_time",
    "tail_mean",
    "quality_score",
    "quality_score_from_trace",
    "quality_score_from_trace_segmented",
    "mc_result_quality",
    "summarize_mc_quality",
]

import numpy as np
from scipy.spatial.transform import Rotation


BODY_BORESIGHT = np.array([0.0, 1.0, 0.0])


def compute_error_trace(
    states: np.ndarray,
    times: np.ndarray,
    q_goal: np.ndarray | None = None,
    boresight_goal: np.ndarray | None = None,
    body_boresight: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-timestep pointing error in degrees.

    For quaternion goals (``q_goal`` provided): geodesic angle to goal.
    For ECI/vector goals (``boresight_goal`` provided): boresight angle.

    Parameters
    ----------
    states : (N, n_state) array
        Simulated states. Columns 3:7 are quaternion [w, x, y, z].
    times : (N,) array
        Time stamps in seconds (or J2000 centuries — only relative values
        matter for settle_time).
    q_goal : (4,) array or None
        Fixed attitude goal quaternion [w, x, y, z].
    boresight_goal : (N, 3) or (3,) array or None
        ECI boresight goal direction per timestep.
    body_boresight : (3,) array or None
        Body-frame boresight axis (default [0, 1, 0]).

    Returns
    -------
    errors : (N,) array
        Pointing error at each timestep in degrees.
    """
    if body_boresight is None:
        body_boresight = BODY_BORESIGHT
    N = len(times)
    errors = np.full(N, 180.0)

    if q_goal is not None:
        qg = q_goal / np.linalg.norm(q_goal)
        for k in range(N):
            q = states[k, 3:7]
            q = q / np.linalg.norm(q)
            cos_half = min(abs(np.dot(q, qg)), 1.0)
            errors[k] = np.degrees(2 * np.arccos(cos_half))
    elif boresight_goal is not None:
        bg = np.atleast_2d(boresight_goal)
        for k in range(N):
            q = states[k, 3:7]
            q = q / np.linalg.norm(q)
            # scipy uses [x, y, z, w] convention
            R_k = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            bore_eci = R_k @ body_boresight
            gk = bg[min(k, len(bg) - 1)]
            gn = np.linalg.norm(gk)
            if gn > 1e-10:
                c = np.clip(np.dot(bore_eci, gk / gn), -1.0, 1.0)
                errors[k] = np.degrees(np.arccos(c))
    return errors


def settle_time(
    times: np.ndarray,
    errors: np.ndarray,
    thresh_deg: float = 5.0,
) -> float:
    """First time error drops below ``thresh_deg`` and stays below.

    Returns time in same units as ``times``. If never settles, returns
    ``times[-1] - times[0]`` (full duration).
    """
    T = times[-1] - times[0]
    for k in range(len(times)):
        if errors[k] < thresh_deg and np.all(errors[k:] < thresh_deg):
            return times[k] - times[0]
    return T


def tail_mean(errors: np.ndarray, frac: float = 0.5) -> float:
    """Mean error over the last ``frac`` fraction of the trajectory (degrees)."""
    start = max(0, int(len(errors) * (1.0 - frac)))
    return float(np.mean(errors[start:]))


def quality_score(
    settle_frac: float,
    tail_mean_deg: float,
) -> float:
    """Composite quality score from pre-computed components.

    score = settle_frac + tail_mean_deg / 180
    """
    return settle_frac + tail_mean_deg / 180.0


def quality_score_from_trace(
    times: np.ndarray,
    errors: np.ndarray,
    settle_thresh_deg: float = 5.0,
    tail_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Compute quality score and its components from an error trace.

    Returns (score, settle_time_s, settle_frac, tail_mean_deg).
    """
    T = times[-1] - times[0]
    st = settle_time(times, errors, settle_thresh_deg)
    sf = st / T if T > 0 else 1.0
    tm = tail_mean(errors, tail_frac)
    return quality_score(sf, tm), st, sf, tm


def quality_score_from_trace_segmented(
    times: np.ndarray,
    errors: np.ndarray,
    boresight_goal: np.ndarray | None = None,
    settle_thresh_deg: float = 5.0,
    tail_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Segment-wise quality score for multi-goal trajectories.

    Identifies goal-change boundaries from ``boresight_goal`` and evaluates
    quality within each active segment (non-zero goal).  Skips idle/transition
    segments where goal is [0, 0, 0].

    The per-segment score uses ``quality_score_from_trace`` with the segment's
    own time window.  The final score is the **mean** across active segments,
    which avoids penalizing inevitable error spikes at goal transitions.

    Falls back to ``quality_score_from_trace`` when ``boresight_goal`` is None
    or there is only one segment.

    Returns (score, mean_settle_time_s, mean_settle_frac, mean_tail_mean_deg).
    """
    if boresight_goal is None or boresight_goal.ndim != 2:
        return quality_score_from_trace(times, errors, settle_thresh_deg, tail_frac)

    # Detect segment boundaries (where goal changes)
    diffs = np.linalg.norm(np.diff(boresight_goal, axis=0), axis=1)
    change_idx = np.where(diffs > 0.01)[0]  # small tolerance for float noise
    segment_bounds = np.split(np.arange(len(times)), change_idx + 1)

    # Filter to active segments (non-zero goal)
    seg_scores = []
    seg_settles = []
    seg_fracs = []
    seg_tails = []
    for seg_idx in segment_bounds:
        if len(seg_idx) < 3:
            continue  # too short to evaluate
        gv = boresight_goal[seg_idx[0]]
        if np.linalg.norm(gv) < 0.01:
            continue  # idle segment, skip

        seg_times = times[seg_idx]
        seg_errors = errors[seg_idx]
        sc, st, sf, tm = quality_score_from_trace(
            seg_times, seg_errors, settle_thresh_deg, tail_frac)
        seg_scores.append(sc)
        seg_settles.append(st)
        seg_fracs.append(sf)
        seg_tails.append(tm)

    if not seg_scores:
        return quality_score_from_trace(times, errors, settle_thresh_deg, tail_frac)

    return (
        float(np.mean(seg_scores)),
        float(np.mean(seg_settles)),
        float(np.mean(seg_fracs)),
        float(np.mean(seg_tails)),
    )


def mc_result_quality(
    result: dict,
    settle_thresh_deg: float = 5.0,
    tail_frac: float = 0.5,
) -> dict:
    """Compute quality metrics for one MC result dict.

    Parameters
    ----------
    result : dict
        MC result with keys: 'state' (N, n_state), 'time' (N,),
        optionally 'q_goal' (4,) or 'boresight_goal' (N, 3).

    Returns
    -------
    dict with keys:
        seed, final_err, settle_time_s, settle_frac, tail_mean,
        quality_score, grade
    """
    states = np.array(result['state'])
    times = np.array(result['time'])
    q_goal = result.get('q_goal', None)
    boresight_goal = result.get('boresight_goal', None)

    if q_goal is not None:
        q_goal = np.array(q_goal)
        if q_goal.ndim > 1:
            q_goal = q_goal[-1]  # use last goal quaternion
    if boresight_goal is not None:
        boresight_goal = np.array(boresight_goal)

    errors = compute_error_trace(states, times, q_goal=q_goal,
                                 boresight_goal=boresight_goal)
    score, st, sf, tm = quality_score_from_trace(
        times, errors, settle_thresh_deg, tail_frac)

    grade = "★★★" if score < 0.3 else ("★★" if score < 0.5 else ("★" if score < 1.0 else "✗"))

    return {
        'seed': result.get('run_id', -1),
        'final_err': errors[-1],
        'settle_time_s': st,
        'settle_frac': sf,
        'tail_mean': tm,
        'quality_score': score,
        'grade': grade,
        'errors': errors,
        'times': times,
        'plan_time_s': result.get('plan_time_s', 0),
        'sim_time_s': result.get('sim_time_s', 0),
    }


def summarize_mc_quality(
    results: list[dict],
    settle_thresh_deg: float = 5.0,
    tail_frac: float = 0.5,
    label: str = "",
) -> dict:
    """Compute summary statistics for a list of MC results.

    Parameters
    ----------
    results : list of dict
        MC result dicts (from pickle 'obj0').
    label : str
        Config label for printing.

    Returns
    -------
    dict with summary stats and list of per-seed quality dicts.
    """
    valid = [r for r in results if r and r.get('traj_valid')]
    metrics = [mc_result_quality(r, settle_thresh_deg, tail_frac) for r in valid]

    if not metrics:
        return {'label': label, 'n': 0, 'metrics': []}

    scores = np.array([m['quality_score'] for m in metrics])
    finals = np.array([m['final_err'] for m in metrics])
    settles = np.array([m['settle_time_s'] for m in metrics])
    tails = np.array([m['tail_mean'] for m in metrics])
    plan_ts = np.array([m['plan_time_s'] for m in metrics])

    summary = {
        'label': label,
        'n': len(metrics),
        'metrics': metrics,
        # Quality score
        'score_mean': float(np.mean(scores)),
        'score_med': float(np.median(scores)),
        'score_p95': float(np.percentile(scores, 95)),
        'score_max': float(np.max(scores)),
        'pct_3star': float(np.mean(scores < 0.3) * 100),
        'pct_2star': float(np.mean(scores < 0.5) * 100),
        'pct_1star': float(np.mean(scores < 1.0) * 100),
        # Final error
        'final_mean': float(np.mean(finals)),
        'final_med': float(np.median(finals)),
        'final_max': float(np.max(finals)),
        'pct_lt1': float(np.mean(finals < 1) * 100),
        'pct_lt5': float(np.mean(finals < 5) * 100),
        # Settle time
        'settle_mean': float(np.mean(settles)),
        'settle_med': float(np.median(settles)),
        # Tail mean
        'tail_mean_mean': float(np.mean(tails)),
        'tail_mean_med': float(np.median(tails)),
        # Plan time
        'plan_mean': float(np.mean(plan_ts)),
        'plan_max': float(np.max(plan_ts)),
    }
    return summary


def print_summary_table(summaries: list[dict]) -> None:
    """Print a comparison table from multiple summarize_mc_quality outputs."""
    if not summaries:
        return
    hdr = (f"{'Config':<25s} {'N':>3s} {'★★★':>5s} {'★★':>5s} "
           f"{'<1°':>5s} {'<5°':>5s} {'Score':>6s} {'Final':>6s} "
           f"{'Settle':>7s} {'Tail50':>7s} {'Plan':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        if s['n'] == 0:
            print(f"{s['label']:<25s}   0")
            continue
        print(f"{s['label']:<25s} {s['n']:3d} "
              f"{s['pct_3star']:4.0f}% {s['pct_2star']:4.0f}% "
              f"{s['pct_lt1']:4.0f}% {s['pct_lt5']:4.0f}% "
              f"{s['score_med']:5.2f}  {s['final_med']:5.1f}° "
              f"{s['settle_med']:6.0f}s {s['tail_mean_med']:5.1f}° "
              f"{s['plan_mean']:5.1f}s")
