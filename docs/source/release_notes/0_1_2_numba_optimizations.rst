0.1.2 Numba optimizations for faster simulation (2026-04-15)
=============================================================

The math helpers were greatly improved! Almost every core function in
``ADCS/helpers/math_helpers.py`` now has a ``@njit(cache=True)``
implementation, so the heavy lifting happens in compiled machine code rather
than interpreted Python. First-call JIT compile time is paid once per session
(or skipped entirely on subsequent runs thanks to the cache), and from there
the speedups are free.

New Features
------------
- Added ``numba``-accelerated paths for the main math helper functions.
- Added dependency: ``numba==0.65.0``.

What Changed Under the Hood
----------------------------

Most of the work was inside ``math_helpers.py``. The key idea is that
``@njit`` functions can't use arbitrary Python objects, so each helper was
rewritten to work purely with NumPy arrays and scalars

Here's a quick tour of what got changed:

**Quaternion algebra** — ``mrp_to_quat``, ``cayley_to_quat``,
``quat_to_mrp``, ``quat_to_cayley``, ``quat_to_vec3``, ``vec3_to_quat``,
``quat_inv``, ``quat_diff``, ``rot_exp``, and ``dcm_to_quat`` all now
pre-allocate output arrays with ``np.empty`` and fill them element-by-element
rather than building intermediate lists. The ``mrp_to_quat`` rewrite is
worth calling out specifically: the old path computed ``atan(||mrp||)`` and
then normalized, which is numerically equivalent but more expensive than the
direct closed-form ``(1 − s^2)/(1 + s^2)`` identity used now.

**Dynamics kernels** — Three new private ``@njit`` helpers handle the
satellite dynamics inner loop:

- ``_quat_qdot(w, q)`` — quaternion kinematics (replaces ``0.5 * w @ Wmat(q).T``)
- ``_wdot_no_rw_kernel(w, total_torque, J, invJ_noRW)`` — angular acceleration without reaction wheels
- ``_wdot_rw_kernel(w, h, total_torque, J, invJ_noRW, rw_axes)`` — angular acceleration with reaction wheels
- ``_rw_hdot_kernel(u_rw, wdot, rw_axes, rw_js)`` — reaction wheel momentum derivative

These are called directly from ``satellite.py``, replacing the inline
``np.cross`` / ``np.diagflat`` expressions that couldn't be JIT-compiled.

**Norm and Jacobian utilities**: ``norm`` now uses a manual ``sqrt(sum of
squares)`` loop (faster under ``njit`` than ``np.linalg.norm``). ``normalize``
pre-allocates its output. The Jacobian helpers ``normed_vec_jac``,
``vec_norm_jac``, and ``vec_norm_hess`` each got a ``@njit`` inner kernel with a thin
Python wrapper to preserve the optional-``dv`` API.

**Quaternion multiply**: A new ``_quat_mult_2`` kernel handles the two-
quaternion case. The public ``quat_mult`` wrapper stays pure Python so it
can still accept variable-length argument lists, but the inner multiply is
now compiled.

**``rot_mat``**: Not JIT-compiled, but its formatting was cleaned up.

**Reaction wheel actuator**: Bias and noise states are now *updated* before their values are *read*, so
the actuator output always reflects the current step rather than the previous
one.

**Benchmark improvements**: ``benchmark_planner.py`` now has a ``--warmup``
CLI flag. Warmup runs trigger Numba's JIT compilation before the
timed window opens, so benchmark numbers reflect steady-state performance
rather than the first-call compilation penalty.

Benchmark Results
-----------------
 
Benchmarks were run on the five main tutorial scripts. The warmup run was
excluded from all timings.
 
.. list-table:: Tutorial Benchmark Comparison
   :widths: 40 20 20 20
   :header-rows: 1
 
   * - Benchmark Script
     - Main (Avg)
     - Numba (Avg)
     - Percent Faster
   * - 01_underactuated_control.py
     - 54.303 s
     - 37.706 s
     - 30.56%
   * - 02_noisy_control.py
     - 31.660 s
     - 31.626 s
     - 0.11%
   * - 03_simple_estimation.py
     - 55.932 s
     - 48.000 s
     - 13.09%
   * - 04_complex_estimation.py
     - 76.758 s
     - 72.910 s
     - 5.01%
   * - 05_orbit_estimation.py
     - 9.002 s
     - 8.407 s
     - 6.61%
   * - **Overall Average**
     - **45.369 s**
     - **39.766 s**
     - **12.35%**