__all__ = ["ADCSLogger"]

import pandas as pd
import numpy as np
from typing import List, Any

class ADCSLogger:
    """
    Log one timestep of data.

    Parameters
    ----------
    t : float
        Simulation time.
    **kwargs :
        Any number of named variables (arrays, objects, etc.)
    """

    def __init__(self, autoconvert_numpy: bool = True, flatten_scalars: bool = True):
        self.records: List[Any] = []
        self.autoconvert_numpy = autoconvert_numpy
        self.flatten_scalars = flatten_scalars

    def log(self, t, **kwargs):
        """
        Log one timestep of data.

        Parameters
        ----------
        t : float
            Simulation time.
        **kwargs :
            Any number of named variables (arrays, objects, etc.)
        """
        entry = {"time": float(t)}
        for name, value in kwargs.items():
            entry[name] = self._serialize_value(value)
        self.records.append(entry)

    def _serialize_value(self, val):
        """Convert supported types into a form suitable for DataFrame storage."""
        # Scalars
        if np.isscalar(val):
            return float(val)

        # NumPy arrays
        if isinstance(val, np.ndarray):
            if self.autoconvert_numpy:
                # Flatten small vectors for readability
                if val.size <= 3 and val.ndim == 1:
                    return tuple(val.tolist())
                else:
                    return val.copy()
            else:
                return val

        # Builtin containers
        if isinstance(val, (list, tuple)):
            return tuple(val)

        # Dataclasses or custom objects
        if hasattr(val, "__dict__"):
            # Store the raw object, not its internals (lazy unpack)
            return val

        # Anything else → fallback
        return str(val)

    def to_dataframe(self):
        """Convert logged data to a pandas DataFrame."""
        return pd.DataFrame(self.records)

    def save(self, path):
        """Save the log to a pickle (for arbitrary objects)."""
        df = self.to_dataframe()
        df.to_pickle(path)

    def load(self, path):
        """Load a previously saved log."""
        self.records = pd.read_pickle(path).to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
        