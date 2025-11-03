import numpy as np
from typing import Sequence
from ADCS.orbits.universal_constants import TimeConstants

class Bias:
    def __init__(self, bias: float = 0.0, std_bias: float = 0.0, bounds: Sequence[float] = (-np.inf, np.inf)) -> None:
        self.bias = bias
        self.std_bias = std_bias
        self.last_bias_time = float('nan')

    def __bool__(self):
        return not (self.bias == 0.0 and self.std_bias == 0.0)

    def _update_bias(self, j2000: float) -> None:
        """Random Walk"""
        if not np.isfinite(self.last_bias_time):
            self.last_bias_time = j2000
            return
        
        dt_centuries = j2000 - self.last_bias_time
        if dt_centuries <= 0:
            return

        dt_sec = dt_centuries * TimeConstants.cent2sec

        self.bias = np.random.normal(
            loc=self.bias,
            scale=self.std_bias * np.sqrt(dt_sec)
        )
        self.bias = np.clip(self.bias, self.bounds[0], self.bounds[1])
        self.last_bias_time = j2000

    def get_bias(self, j2000: float) -> float:
        self._update_bias(j2000=j2000)
        return self.bias