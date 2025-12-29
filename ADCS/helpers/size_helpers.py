__all__ = ["check_numpy_size"]

import numpy as np
from typing import Optional

def check_numpy_size(
    arr: np.ndarray, 
    rows: Optional[int] = None, 
    cols: Optional[int] = None, 
    depth: Optional[int] = None
) -> None:
    """
    Checks the dimensions of a NumPy array against provided optional constraints.
    
    Assumes standard convention:
    - Rows = Axis 0
    - Cols = Axis 1
    - Depth = Axis 2
    
    Args:
        arr: The input numpy array to check.
        rows: Expected number of rows (axis 0).
        cols: Expected number of columns (axis 1).
        depth: Expected depth/channels (axis 2).
        
    Raises:
        AssertionError: If dimensions are not met or if array lacks sufficient dimensions.
    """
    shape = arr.shape
    ndim = arr.ndim

    # Helper to generate clear error messages
    def error_msg(dim_name, axis, expected, actual):
        return (f"Expected {expected} {dim_name} (axis {axis}), "
                f"but got {actual}. Full shape is {shape}.")

    # 1. Check Rows (Axis 0)
    if rows is not None:
        if ndim < 1:
            raise AssertionError(f"Cannot check rows on 0-d array. Shape is {shape}.")
        if shape[0] != rows:
            raise AssertionError(error_msg("rows", 0, rows, shape[0]))

    # 2. Check Columns (Axis 1)
    if cols is not None:
        if ndim < 2:
            raise AssertionError(f"Cannot check cols on array with < 2 dimensions. Shape is {shape}.")
        if shape[1] != cols:
            raise AssertionError(error_msg("cols", 1, cols, shape[1]))

    # 3. Check Depth (Axis 2)
    if depth is not None:
        if ndim < 3:
            raise AssertionError(f"Cannot check depth on array with < 3 dimensions. Shape is {shape}.")
        if shape[2] != depth:
            raise AssertionError(error_msg("depth", 2, depth, shape[2]))