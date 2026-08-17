:orphan:

.. _ssc26:

.. include:: _analytics.txt

==========================================================
Test any published attitude control law on your bus.
==========================================================

.. rst-class:: ssc26-lede

   **Without reimplementing it.**

*Generalized Attitude Control for Small Spacecraft* — SmallSat 2026,
poster SSC26-P2-54.

You scanned the poster. Everything below runs against the shipped library.

Install
=======

.. code-block:: console

   pip install generalized-adcs

Same line as the poster. Python 3.10+.

A published magnetic law gets a wheel
=====================================

The Lovera–Astolfi law is magnetorquer-only: it projects its desired torque
onto the plane perpendicular to **B** and never commands a wheel. Change the
*allocator*, not the law, and the same feedback starts using the wheel:

.. code-block:: python

   law = Lovera_Law(J=sat.J_0, kp=2e-5, kd=2e-2)  # unmodified
   mtq_only = PipelineController(sat, law)        # published
   with_wheel = PipelineController(sat, law,      # + the wheel
       alloc_config=AllocationConfig(method='lp'))

.. note::

   This is the ``ADCS.pipeline`` adapter, which is on its way into ``main``.
   The results below are measured; the code lands with it. Everything under
   "What is available today" works in the released package right now.

``mtq_only`` reproduces the published ``MTQ_Lovera`` controller to machine
precision — max :math:`|\Delta u|` = 2.2e-16 over 200 random states, 122 of
them bit-identical. ``with_wheel`` is the same law on the same bus, now
commanding the wheel, because goal formulation, compensation and allocation
are separate stages around it.

On 3MTQ+1RW over 100 paired trials, that one change takes Lovera from **0% to
29%** convergence on full attitude and **41% to 80%** on vector pointing;
Wisniewski goes from 2% to 39% and 16% to 72% (paper Table 7).

.. _ssc26-how-it-works:

How it works
============

The adapter is a five-stage pipeline. A control law is Stage 2; everything
around it adapts the law to the hardware you actually have.

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Stage
     - Block
     - What it does
   * - 1
     - Goal formulation
     - Converts any goal (full attitude, pointing vector, none) into the error
       signals the law declares it wants, plus the rate projector ``P``.
   * - 2
     - **Control law**
     - *Your code.* Maps error signals to a desired body torque.
   * - 3
     - Interface
     - Adapts laws that emit actuator commands rather than torque.
   * - 4
     - Compensation
     - Gyroscopic, frame-rotation, disturbance feedforward, damping injection —
       each **skipped** if the law says it does that term itself.
   * - 5
     - Allocation
     - LP / QP / weighted-QP / pseudoinverse / cross-product, plus momentum
       management, onto the actual actuator set.

Working code
------------

Every code block from the poster is executed verbatim by
``papers/SSC26_poster/verify_snippets.py``, so what is printed is what runs.

.. note::

   The five-stage adapter shown above ships in the ``ADCS.pipeline`` package,
   which is on its way into ``main``. Until it lands, this page describes the
   architecture and the paper reports the results; the runnable examples are in
   the repository. See :doc:`code` for where each stage lives.

What is available today
-----------------------

The released package already carries the control laws, the LP/QP allocators,
magnetorquer and reaction-wheel models, UKF-family estimators, and IGRF-13 with
Skyfield frames:

.. code-block:: python

   import numpy as np
   import ADCS

   acts: list[ADCS.Actuator] = [
       ADCS.RW(axis=np.array([0, 0, 1.0]), max_torque=0.0023,
               J=5.7e-6, h=0.0, h_max=0.0036)
   ]
   acts += [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
   sens = [ADCS.MTM(axis) for axis in np.eye(3)]
   sat = ADCS.Satellite(mass=4.0, J_0=np.diag([0.03, 0.03, 0.01]),
                        actuators=acts, sensors=sens,
                        boresight=np.array([0, 0, 1.0]))

Paper and citation
==================

.. TODO(ssc26): replace with the final SSC26-P2-54 PDF URL once published.

P. McKeen, N. Scheuer and K. Cahoy, *Generalized Attitude Control for Small
Spacecraft*, SSC26-P2-54, 40th Annual Small Satellite Conference, Logan UT,
August 2026.

.. code-block:: bibtex

   @inproceedings{mckeen2026generalized,
     title     = {Generalized Attitude Control for Small Spacecraft},
     author    = {McKeen, Patrick and Scheuer, Niclas and Cahoy, Kerri},
     booktitle = {Proceedings of the 40th Annual Small Satellite Conference},
     number    = {SSC26-P2-54},
     year      = {2026},
     address   = {Logan, UT},
   }

Where to go next
================

- :doc:`run` — run it in your browser, no install
- :doc:`paper` — the paper and how to cite it
- :doc:`code` — the repository, and where each stage lives
- :doc:`contact` — questions, collaboration, bug reports
- :doc:`../installation/index` — full installation guide
- :doc:`../tutorials/index` — tutorials
