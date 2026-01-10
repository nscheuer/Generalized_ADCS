__all__ = ["MonteCarloRunner"]

import numpy as np
import concurrent.futures
from tqdm import tqdm
from typing import Callable, List, Dict, Any
import multiprocessing

class MonteCarloRunner:
    def __init__(self, sim_func: Callable[[Dict[str, Any]], Dict[str, Any]], config_generator: Callable[[int], Dict[str, Any]], num_runs: int, max_workers: int = None) -> None:
        self.sim_func = sim_func
        self.config_generator = config_generator
        self.num_runs = num_runs
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()

    def run(self) -> List[Dict[str, Any]]:
        configs = [self.config_generator(i) for i in range(self.num_runs)]

        results = []

        print(f"Starting Monte Carlo: {self.num_runs} runs on {self.max_workers} cores.")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(self.sim_func, config): i for i, config in enumerate(configs)}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=self.num_runs, desc="MC Progress"):
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    print(f"Run generated an exception: {e}")
                    results.append({"status": "failed", "error": str(e)})
                    
        return results