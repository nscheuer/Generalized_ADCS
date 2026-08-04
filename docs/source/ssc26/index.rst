:orphan:

.. _ssc26:

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

   pip install git+https://github.com/nscheuer/Generalized_ADCS.git

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
   ctrl = PipelineController(sat, MyLaw(),      # steps 2-3
       alloc_config=AllocationConfig(method='lp'))

No double-counting
------------------

A law that already performs its own gyroscopic term declares it, and Stage 4
skips that term rather than adding it twice:

.. code-block:: python

   class Lovera(ControlLaw):
       # law does its own w x (Jw + h), so
       # Stage 4 must not add it again:
       interface = LawInterface(includes_gyroscopic=True)

This is not just a configuration flag: hand-forcing gyroscopic compensation
around such a law demonstrably changes the output, and the guard is what
prevents it (``testing/test_pipeline/test_lovera_law.py``).

Goal type as a design lever
---------------------------

``Attitude_Goal`` and ``Vector_Goal`` are the abstract interfaces; use a
concrete goal such as ``Fixed_Attitude_Goal`` or ``ECI_Goal``:

.. code-block:: python

   full = Fixed_Attitude_Goal(q_tgt)   # 49% converge
   vec  = ECI_Goal(u_tgt)              # 83% converge
   u_full = ctrl.find_u(x, sens, sat, os_now, full)
   u_vec  = ctrl.find_u(x, sens, sat, os_now, vec)
   # same law, same bus - Stage 1 converts each goal

Same law, same hardware — relaxing an over-specified full-attitude goal to the
pointing the mission actually needs is often worth more than more actuators
(paper Table 7).

Swap the allocator
------------------

.. code-block:: python

   AllocationConfig(method='lp')   # direction kept
   AllocationConfig(method='qp')   # size kept, tilts
   AllocationConfig(method='magnetic_cross')  # MTQ only

Inside the achievable torque polytope every allocator returns the request, so
LP and QP differ only under saturation. There LP holds direction to
:math:`0.00^\circ` and gives up magnitude; QP recovers roughly 50% more
magnitude at up to :math:`26.9^\circ` of tilt. On 3MTQ+1RW full attitude the
LP wins (**42% vs 39%**); on a magnetorquer-only bus the QP does (paper
Table 6, §IV-F).

Every block on this page is executed verbatim by
``papers/SSC26_poster/verify_snippets.py`` in CI, so what is printed is what
runs.

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

- :doc:`code` — the repository, and where each stage lives
- :doc:`contact` — questions, collaboration, bug reports
- :doc:`../installation/index` — full installation guide
- :doc:`../tutorials/index` — tutorials
