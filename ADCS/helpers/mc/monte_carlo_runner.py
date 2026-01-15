import multiprocessing
import queue
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Dict, Any, Optional

# --- RICH IMPORTS ---
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.console import Group

# --- GLOBAL WORKER STATE ---
# These exist only within the worker processes
_WORKER_PROGRESS_Q: Optional[multiprocessing.Queue] = None
_WORKER_SLOT_Q: Optional[multiprocessing.Queue] = None

def _worker_init(progress_q, slot_q):
    """Internal initializer for worker processes."""
    global _WORKER_PROGRESS_Q, _WORKER_SLOT_Q
    _WORKER_PROGRESS_Q = progress_q
    _WORKER_SLOT_Q = slot_q

# --- PUBLIC WORKER API ---
# Use these functions inside your simulation worker

def claim_worker_slot() -> int:
    """Claims a specific UI row (slot) for this worker."""
    if _WORKER_SLOT_Q is None:
        return -1
    return _WORKER_SLOT_Q.get()

def release_worker_slot(slot_id: int) -> None:
    """Releases the UI row so another job can use it."""
    if _WORKER_SLOT_Q is not None and slot_id != -1:
        _WORKER_SLOT_Q.put(slot_id)

def update_worker_progress(slot_id: int, run_id: int, step: int, total: int) -> None:
    """Sends a progress update to the main dashboard."""
    if _WORKER_PROGRESS_Q is not None and slot_id != -1:
        _WORKER_PROGRESS_Q.put((slot_id, run_id, step, total))

# --- MAIN RUNNER CLASS ---

class MonteCarloRunner:
    def __init__(self, 
                 sim_func: Callable[[Dict[str, Any]], Dict[str, Any]], 
                 config_generator: Callable[[int], Dict[str, Any]], 
                 num_runs: int, 
                 max_workers: int = None):
        self.sim_func = sim_func
        self.config_generator = config_generator
        self.num_runs = num_runs
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()

    def run(self) -> List[Dict[str, Any]]:
        # 1. Setup IPC (Manager)
        manager = multiprocessing.Manager()
        progress_q = manager.Queue()
        slot_q = manager.Queue()
        
        # Pre-fill slots (0 to N-1)
        for i in range(self.max_workers):
            slot_q.put(i)

        configs = [self.config_generator(i) for i in range(self.num_runs)]
        results = []

        # 2. Setup Rich Dashboard
        # Per-core progress bars
        job_progress = Progress(
            "{task.description}",
            BarColumn(bar_width=None),
            "{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True
        )
        
        # Map slot_id (0..15) to Rich Task IDs
        slot_task_ids = {} 
        for i in range(self.max_workers):
            tid = job_progress.add_task(f"[dim]Core {i} Idle[/dim]", total=100, visible=False)
            slot_task_ids[i] = tid

        # Overall progress bar
        overall_progress = Progress(
            TextColumn("[bold blue]Total Progress"),
            BarColumn(bar_width=40),
            "{task.completed}/{task.total}",
            TimeRemainingColumn()
        )
        overall_task = overall_progress.add_task("Campaign", total=self.num_runs)

        # Create Layout Group
        dashboard = Group(
            Panel(job_progress, title=f"Active Cores ({self.max_workers})", border_style="blue"),
            overall_progress
        )

        print(f"Starting Monte Carlo: {self.num_runs} runs on {self.max_workers} cores.")

        # 3. Execution Loop
        with Live(dashboard, refresh_per_second=10):
            with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_worker_init, initargs=(progress_q, slot_q)) as executor:
                
                # Submit jobs
                futures = {executor.submit(self.sim_func, cfg): cfg["run_id"] for cfg in configs}
                
                completed_count = 0
                while completed_count < self.num_runs:
                    # A. Drain Progress Queue (Non-blocking)
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

                    # B. Check for Completed Futures
                    # (We manually poll futures here to update the 'Overall' bar)
                    done_futures = [f for f in futures if f.done()]
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