__all__ = ["Trajectory"]

import numpy as np
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

class Trajectory:
    def __init__(self, t: np.ndarray, x: np.ndarray, u: np.ndarray, K: np.ndarray, S: np.ndarray) -> None:
        self.times = t
        self.states = x
        self.controls = u
        self.gains = K
        self.costs = S

        self.start_time = t[0]
        self.end_time = t[-1]
        self.n_steps = len(t)

        self.state_dim = x.shape[0]
        self.ctrl_dim = u.shape[0]

    def is_valid_time(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time
    
    def get_state_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        alpha = (t - self.times[idx]) / (self.times[idx+1] - self.times[idx])
        x0 = self.states[:, idx]
        x1 = self.states[:, idx+1]

        return (1 - alpha) * x0 + alpha * x1
    
    def get_control_at(self, t: float) -> np.ndarray:
        idx = self._get_idx(t)
        alpha = (t - self.times[idx]) / (self.times[idx+1] - self.times[idx])
        return (1 - alpha) * self.controls[:, idx] + alpha * self.controls[:, idx+1]
    
    def get_gain_at(self, t: float) -> np.ndarray:
        idx = (np.abs(self.times - t)).argmin()
        k_flat = self.gains[:, idx]
        return k_flat.reshape(self.ctrl_dim, self.state_dim)
    
    def compute_tracking_control(self, t: float, x_current: np.ndarray) -> np.ndarray:
        if not self.is_valid_time(t):
            raise ValueError(f"Time {t} is outside trajectory bounds [{self.start_time}, {self.end_time}]")
        
        x_ref = self.get_state_at(t)
        u_ref = self.get_control_at(t)
        K = self.get_gain_at(t)
        
        dx = x_current - x_ref
        if self.state_dim >= 7:
            if np.dot(x_current[3:7], x_ref[3:7]) < 0:
                dx[3:7] = x_current[3:7] + x_ref[3:7] 

        return u_ref - K @ dx

    def get_plotting_data(self) -> Dict[str, np.ndarray]:
        return {
            "time": self.times,
            "state": self.states,
            "control": self.controls,
            "cost": self.costs
        }
    
    def _get_idx(self, t: float) -> int:
        if t >= self.end_time: 
            return self.n_steps - 2
        idx = np.searchsorted(self.times, t, side='right') - 1
        return max(0, min(idx, self.n_steps - 2))
