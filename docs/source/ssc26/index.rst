:orphan:

.. _ssc26:

===================================================
Generalized Attitude Control for Small Spacecraft
===================================================

.. rst-class:: ssc26-lede

   **Any published attitude control law. Any actuator set. One adapter.**

.. admonition:: SmallSat 2026 · Poster SSC26-P2-54
   :class: tip

   You scanned the poster. Everything below runs against the shipped library.

Install
=======

.. code-block:: console

   pip install git+https://github.com/nscheuer/Generalized_ADCS.git

A magnetic-only law gets a reaction wheel
=========================================

The published Lovera–Astolfi law is magnetorquer-only: it projects its desired
torque onto the plane perpendicular to **B** and never commands a wheel. Change
the *allocator*, not the law, and the same control law starts using the wheel:

.. code-block:: python

   law = PD_Law(kp=2e-5, kd=2e-2, eps=1.0)   # unmodified
   mtq_only = PipelineController(sat, law)   # MTQ only
   with_wheel = PipelineController(sat, law, # Stage 5 swap
       alloc_config=AllocationConfig(method='lp'))

``mtq_only`` reproduces the published ``MTQ_Lovera`` controller to machine
precision (max :math:`|\Delta u|` = 2.2e-16). ``with_wheel`` is the same law on
the same bus, now commanding the wheel — because goal formulation, compensation
and allocation are separate stages around it.

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

Bring your own law
------------------

Implement one method. Declare what error signals you want, and the rest of the
pipeline reconfigures around you:

.. code-block:: python

   class MyLaw(ControlLaw):
       interface = LawInterface()   # full attitude + rate
       kp, kd = 2e-5, 2e-2

       def compute(self, q_err, w_err=None, **kw):
           return -(self.kp * q_err + self.kd * w_err)

No double-counting
------------------

A law that already includes its own gyroscopic term declares it, and Stage 4
skips that term rather than adding it twice:

.. code-block:: python

   interface = LawInterface(includes_gyroscopic=True)
   # -> CompensationConfig.enable_gyroscopic is False

Goal type as a design lever
---------------------------

.. code-block:: python

   ctrl = PipelineController(sat, law,
       alloc_config=AllocationConfig(method='lp'))
   star = ECI_Goal(np.array([-0.14, -0.37, -0.92]))
   u_eci = ctrl.find_u(x, sens, sat, os_now, star)
   u_nadir = ctrl.find_u(x, sens, sat, os_now, Nadir_Goal())
   # same law, same bus - Stage 1 converts each goal

Swap the allocator
------------------

.. code-block:: python

   AllocationConfig(method='lp')   # direction kept
   AllocationConfig(method='qp')   # size kept, tilts
   AllocationConfig(method='qpw')  # perp error x100
   AllocationConfig(method='magnetic_cross')  # MTQ only

Every block on this page is executed verbatim by
``papers/SSC26_poster/verify_snippets.py`` in CI, so what is printed is what
runs.

Paper and citation
==================

.. TODO(ssc26): replace with the final SSC26-P2-54 PDF URL once published.

*Generalized Attitude Control for Small Spacecraft*, SSC26-P2-54,
40th Annual Small Satellite Conference, Logan UT, August 2026.

.. code-block:: bibtex

   @inproceedings{mckeen2026generalized,
     title     = {Generalized Attitude Control for Small Spacecraft},
     author    = {McKeen, Patrick and Scheuer, Niclas},
     booktitle = {Proceedings of the 40th Annual Small Satellite Conference},
     number    = {SSC26-P2-54},
     year      = {2026},
     address   = {Logan, UT},
   }

Where to go next
================

- :doc:`code` — the repository, and where each stage lives
- :doc:`contact` — questions, collaboration, bug reports
- :doc:`../installation/index` — full installation guide
- :doc:`../tutorials/index` — tutorials
