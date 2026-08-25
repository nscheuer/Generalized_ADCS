:orphan:

.. _ssc26-paper:

.. include:: _analytics.txt

==================
SSC26 · The paper
==================

**Generalized Attitude Control for Small Spacecraft**

Patrick McKeen\ :sup:`1`, Niclas Scheuer\ :sup:`2`, Kerri Cahoy\ :sup:`1`

:sup:`1` MIT Department of Aeronautics and Astronautics ·
:sup:`2` ETH Zürich Department of Mechanical and Process Engineering

Paper **SSC26-P2-54**, 40th Annual Small Satellite Conference, Salt Lake City UT,
August 2026.

Read it
=======

- **Paper** — https://digitalcommons.usu.edu/smallsat/2026/all2026/253

  The proceedings entry on DigitalCommons@USU. The PDF is linked from that page
  under *Additional Files*; the SmallSat collection is open access, so no
  institutional login is needed.

- **Ask the authors** — :doc:`contact`, or pmckeen@mit.edu

Presented in **Poster Session 2**, Salt Palace Convention Center, Salt Lake City UT.

Abstract
========

Many classical attitude-control designs assume unconstrained torque or are
developed for a fixed actuator set, requiring redesign for each new spacecraft
and its goals. We present a framework that treats the control law as a black
box and places goal formulation, compensation, torque allocation, and momentum
management in a common adapter around it, so a law from the literature can be
adapted across heterogeneous and underactuated actuator sets without
reimplementation.

Before any simulation, a controllability analysis screens whether a given
actuator set can meet a goal. Over a full orbit the geomagnetic field sweeps
the magnetorquer torque envelope through all three axes, so magnetorquers alone
are controllable for full attitude in the orbit-averaged sense. Reduced-attitude
objectives such as camera-boresight pointing relax the requirement further.

The framework is validated in 100-trial Monte Carlo campaigns across
configurations from magnetorquer-only to three-wheel, including the 3MTQ+1RW
hybrid. Because goal, law, and allocation are separated, a shortfall can be
charged to hardware geometry, control law, or allocator strategy rather than to
the controller as a whole.

Cite it
=======

.. code-block:: bibtex

   @inproceedings{mckeen2026generalized,
     title     = {Generalized Attitude Control for Small Spacecraft},
     author    = {McKeen, Patrick and Scheuer, Niclas and Cahoy, Kerri},
     booktitle = {Proceedings of the 40th Annual Small Satellite Conference},
     number    = {SSC26-P2-54},
     year      = {2026},
     address   = {Salt Lake City, UT},
     url       = {https://digitalcommons.usu.edu/smallsat/2026/all2026/253},
   }

Also on the poster
==================

- :doc:`index` — what the adapter is, and the code
- :doc:`run` — run it in your browser
- :doc:`code` — the repository
