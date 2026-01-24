__all__ = ["check_numpy_size"]

import numpy as np
from typing import Optional

def check_numpy_size(
    arr: np.ndarray, 
    rows: Optional[int] = None, 
    cols: Optional[int] = None, 
    depth: Optional[int] = None
) -> None:
    r"""
    Validate the dimensionality of a NumPy array against expected size constraints.

    This utility enforces structural assumptions on NumPy arrays by checking
    selected axes against expected sizes. It follows the conventional axis
    interpretation:

    .. math::

        \begin{aligned}
        \text{Axis }0 &\rightarrow \text{Rows} \\
        \text{Axis }1 &\rightarrow \text{Columns} \\
        \text{Axis }2 &\rightarrow \text{Depth / Channels}
        \end{aligned}

    Let :math:`\mathbf{A} \in \mathbb{R}^{d_0 \times d_1 \times d_2 \times \dots}`
    denote the input array. For each non-``None`` constraint, the function asserts

    .. math::

        d_i = n_i

    where :math:`n_i` is the expected size along axis :math:`i`.

    This function performs **no reshaping or casting**; it is intended purely
    as a defensive programming and validation tool, commonly used in numerical
    pipelines, estimation routines, and control algorithms to guarantee
    dimensional consistency.

    If a requested axis does not exist (i.e., the array has insufficient
    dimensions), or if the size does not match, an :class:`AssertionError`
    is raised with a detailed diagnostic message.

    :param arr: Input NumPy array whose shape is to be validated.
    :type arr: numpy.ndarray
    :param rows: Expected number of rows (size of axis 0). If ``None``, axis 0 is not checked.
    :type rows: int or None
    :param cols: Expected number of columns (size of axis 1). If ``None``, axis 1 is not checked.
    :type cols: int or None
    :param depth: Expected depth or channel count (size of axis 2). If ``None``, axis 2 is not checked.
    :type depth: int or None
    :return: ``None``. The function succeeds silently if all checks pass.
    :rtype: None

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