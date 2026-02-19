__all__ = ["MonteCarloRunner", "claim_worker_slot", "release_worker_slot", "update_worker_progress"]

import multiprocessing
import queue
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Dict, Any, Optional

from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.console import Group

_WORKER_PROGRESS_Q: Optional[multiprocessing.Queue] = None
_WORKER_SLOT_Q: Optional[multiprocessing.Queue] = None

def _worker_init(progress_q, slot_q):
    r"""
    Initialize per-process global state for worker processes.

    This initializer is executed **once per worker process** when the
    :class:`concurrent.futures.ProcessPoolExecutor` starts. It binds
    inter-process communication (IPC) queues to global variables local
    to each worker, enabling progress reporting and UI slot coordination.

    Let :math:`Q_p` denote the progress queue and :math:`Q_s` the slot queue.
    Each worker process stores references such that

    .. math::

        \forall \text{worker } w:\quad
        (Q_p^{(w)}, Q_s^{(w)}) = (Q_p, Q_s)

    This allows all workers to emit progress updates into a shared stream
    while mutually excluding UI slots via a token-based allocation.

    :param progress_q: Shared queue for sending progress updates to the main process.
    :type progress_q: multiprocessing.Queue
    :param slot_q: Shared queue managing available UI slot identifiers.
    :type slot_q: multiprocessing.Queue

    :return: None
    :rtype: None

    """
    global _WORKER_PROGRESS_Q, _WORKER_SLOT_Q
    _WORKER_PROGRESS_Q = progress_q
    _WORKER_SLOT_Q = slot_q

def claim_worker_slot() -> int:
    r"""
    Claim an exclusive UI slot identifier for the calling worker.

    Each worker retrieves a unique integer token from the shared slot queue
    :math:`Q_s`. This token corresponds to a dedicated row in the live Rich
    dashboard and remains reserved until explicitly released.

    Formally, let :math:`S = \{0, 1, \dots, N-1\}` be the set of available slots.
    Calling this function performs:

    .. math::

        s \leftarrow \operatorname{pop}(Q_s), \quad s \in S

    If the function is invoked outside a worker context (i.e., the global
    queue is uninitialized), the sentinel value :math:`-1` is returned.

    :return: Allocated slot identifier, or ``-1`` if unavailable.
    :rtype: int

    """
    if _WORKER_SLOT_Q is None:
        return -1
    return _WORKER_SLOT_Q.get()

def release_worker_slot(slot_id: int) -> None:
    r"""
    Release a previously claimed UI slot back to the pool.

    This function returns the given slot identifier to the shared slot queue
    :math:`Q_s`, making it available for reuse by another worker. The operation
    is idempotent for invalid slot identifiers.

    In queue-theoretic terms, this performs:

    .. math::

        Q_s \leftarrow Q_s \cup \{s\}

    where :math:`s` is the released slot identifier.

    :param slot_id: Slot identifier obtained from :func:`~claim_worker_slot`.
    :type slot_id: int

    :return: None
    :rtype: None

    """
    if _WORKER_SLOT_Q is not None and slot_id != -1:
        _WORKER_SLOT_Q.put(slot_id)

def update_worker_progress(slot_id: int, run_id: int, step: int, total: int) -> None:
    r"""
    Send a progress update from a worker to the main dashboard.

    This function emits a tuple describing the current simulation state into
    the shared progress queue :math:`Q_p`. The main process consumes these
    updates to refresh the corresponding Rich progress bar.

    Each update is modeled as a discrete progress signal:

    .. math::

        u = (s, r, k, K)

    where

    +------+-----------------------------+
    | Term | Meaning                     |
    +======+=============================+
    | :math:`s` | UI slot identifier     |
    +------+-----------------------------+
    | :math:`r` | Monte Carlo run index  |
    +------+-----------------------------+
    | :math:`k` | Current step           |
    +------+-----------------------------+
    | :math:`K` | Total number of steps  |
    +------+-----------------------------+

    Updates with an invalid slot identifier (``-1``) are silently ignored.

    :param slot_id: UI slot assigned to the worker.
    :type slot_id: int
    :param run_id: Unique Monte Carlo run identifier.
    :type run_id: int
    :param step: Current simulation step.
    :type step: int
    :param total: Total number of steps in the simulation.
    :type total: int

    :return: None
    :rtype: None

    """
    if _WORKER_PROGRESS_Q is not None and slot_id != -1:
        _WORKER_PROGRESS_Q.put((slot_id, run_id, step, total))

class MonteCarloRunner:
    r"""
    Parallel Monte Carlo simulation orchestrator with live Rich dashboard.

    This class manages the execution of :math:`N` independent Monte Carlo
    simulations across multiple CPU cores using
    :class:`concurrent.futures.ProcessPoolExecutor`. It provides real-time
    visualization of per-core progress and global completion status.

    Let :math:`f(\theta_i)` denote a simulation function evaluated at
    configuration :math:`\theta_i`. The runner computes:

    .. math::

        \mathcal{R} = \left\{ f(\theta_1), f(\theta_2), \dots, f(\theta_N) \right\}

    subject to a maximum parallelism constraint :math:`P`, where:

    .. math::

        P = \min(\text{CPU cores}, \text{max\_workers})

    Progress is streamed asynchronously from workers to the main process via
    multiprocessing queues, ensuring minimal synchronization overhead.

    :param sim_func: Callable implementing a single Monte Carlo simulation.
    :type sim_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    :param config_generator: Function mapping run indices to configuration dictionaries.
    :type config_generator: Callable[[int], Dict[str, Any]]
    :param num_runs: Total number of Monte Carlo runs.
    :type num_runs: int
    :param max_workers: Maximum number of worker processes.
    :type max_workers: int

    """
    def __init__(self, 
                 sim_func: Callable[[Dict[str, Any]], Dict[str, Any]], 
                 config_generator: Callable[[int], Dict[str, Any]], 
                 num_runs: int, 
                 max_workers: int = None,
                 per_run_timeout: float = None):
        self.sim_func = sim_func
        self.config_generator = config_generator
        self.num_runs = num_runs
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        self.per_run_timeout = per_run_timeout  # seconds; None = no timeout

    def run(self) -> List[Dict[str, Any]]:
        r"""
        Execute the Monte Carlo campaign and render the live dashboard.

        This method performs the full simulation lifecycle:

        1. Initialize inter-process communication queues.
        2. Allocate UI slots corresponding to worker cores.
        3. Submit simulation jobs to the process pool.
        4. Continuously poll worker progress and completed futures.
        5. Update per-core and global progress bars in real time.

        The execution proceeds until all :math:`N` simulations complete, producing
        a result list:

        .. math::

            \mathcal{R} = \left[ r_1, r_2, \dots, r_N \right]

        where each :math:`r_i` is the return value of ``sim_func`` for run :math:`i`.
        Failed runs yield ``None`` entries.

        The overall progress bar advances according to:

        .. math::

            C(t) = \sum_{i=1}^{N} \mathbf{1}_{\{f_i \text{ completed at } t\}}

        where :math:`C(t)` is the completed run count at time :math:`t`.

        :return: List of simulation results in submission order.
        :rtype: List[Dict[str, Any]]

        """
        manager = multiprocessing.Manager()
        progress_q = manager.Queue()
        slot_q = manager.Queue()
        
        for i in range(self.max_workers):
            slot_q.put(i)

        configs = [self.config_generator(i) for i in range(self.num_runs)]
        results = []

        job_progress = Progress(
            "{task.description}",
            BarColumn(bar_width=None),
            "{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True
        )

        slot_task_ids = {} 
        for i in range(self.max_workers):
            tid = job_progress.add_task(f"[dim]Core {i} Idle[/dim]", total=100, visible=False)
            slot_task_ids[i] = tid

        overall_progress = Progress(
            TextColumn("[bold blue]Total Progress"),
            BarColumn(bar_width=40),
            "{task.completed}/{task.total}",
            TimeRemainingColumn()
        )
        overall_task = overall_progress.add_task("Campaign", total=self.num_runs)

        dashboard = Group(
            Panel(job_progress, title=f"Active Cores ({self.max_workers})", border_style="blue"),
            overall_progress
        )

        print(f"Starting Monte Carlo: {self.num_runs} runs on {self.max_workers} cores.")

        with Live(dashboard, refresh_per_second=10):
            with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_worker_init, initargs=(progress_q, slot_q)) as executor:
                
                futures = {}
                for cfg in configs:
                    f = executor.submit(self.sim_func, cfg)
                    f._start_time = time.time()
                    futures[f] = cfg["run_id"]
                
                completed_count = 0
                while completed_count < self.num_runs:
                    while not progress_q.empty():
                        try:
                            slot, run_id, step, total_steps = progress_q.get_nowait()
                            if slot != -1:
                                task_id = slot_task_ids[slot]
                                job_progress.update(
                                    task_id, 
                                    completed=step, 
                                    total=total_steps, 
                                    visible=True,
                                    description=f"[bold green]Run {run_id}[/]"
                                )
                        except queue.Empty:
                            break

                    done_futures = [f for f in futures if f.done()]
                    
                    # Check for timed-out futures
                    if self.per_run_timeout is not None:
                        now = time.time()
                        for f in list(futures):
                            if not f.done() and hasattr(f, '_start_time'):
                                elapsed = now - f._start_time
                                if elapsed > self.per_run_timeout:
                                    run_id = futures[f]
                                    f.cancel()
                                    print(f"\n[!] Run {run_id} timed out after {elapsed:.0f}s")
                                    results.append({"run_id": run_id, "error": "timeout", "traj_valid": False})
                                    del futures[f]
                                    completed_count += 1
                                    overall_progress.update(overall_task, advance=1)

                    for f in done_futures:
                        try:
                            results.append(f.result())
                        except Exception as e:
                            print(f"\n[!] Error in run: {e}")
                            results.append(None)
                        
                        del futures[f]
                        completed_count += 1
                        overall_progress.update(overall_task, advance=1)
                    
                    time.sleep(0.05)

        return results