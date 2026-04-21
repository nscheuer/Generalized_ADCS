0.1.4 More Numba optimizations and 4x simulation speedup (2026-04-21)
=====================================================================

We made another round of performance-focused improvements to the simulation
core, including tighter hot-loop kernels, reduced temporary allocations, and
cleaner data flow between attitude dynamics and actuator updates.

Result: depending on the scenario, simulations now run up to about **4x
faster** compared to the previous baseline.

Highlights
----------
- Further optimized Numba-compiled math and dynamics paths.
- Reduced Python overhead in per-step simulation loops.
- Improved memory behavior by avoiding unnecessary intermediate arrays.

Speed Comparison
----------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Before
     - After
   * - .. image:: ../_static/release_notes/0_1_4_old_speed.png
          :alt: Terminal output before latest Numba optimizations
          :width: 100%
     - .. image:: ../_static/release_notes/0_1_4_new_speed.png
          :alt: Terminal output after latest Numba optimizations
          :width: 100%
