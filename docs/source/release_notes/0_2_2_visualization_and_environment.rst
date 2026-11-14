0.2.2 Visualization, Environment, and Force Modeling
=====================================================

Animations
----------

The new Godot viewer turns simulation results into an interactive 3-D scene
with spacecraft attitude, orbital motion, pointing targets, and boresight
overlays. The animation below follows ICESat-2 through nadir, coordinate, and
along-track pointing as it passes over Earth at sunset. It was recorded at
10 times simulation speed. A tutorial on creating and recording these
animations is forthcoming. See the `graphics pull request
<https://github.com/nscheuer/Generalized_ADCS/pull/250>`__ for the viewer work.

.. image:: ../_static/release_notes/0_2_2_animation.gif
   :alt: ICESat-2 performing nadir, coordinate, and along-track pointing maneuvers over Earth at sunset.
   :width: 800px
   :align: center

Tutorials
---------

The tutorials are being refreshed to use the updated ``simulate()`` frontend
and to cover the new capabilities in this release. This section will be
expanded as those walkthroughs are added; the existing :doc:`tutorial series
<../tutorials/index>` remains available.

Environmental models
--------------------

The configurable ``Environment`` now offers WMM2025 and CHAOS-8 magnetic
fields alongside IGRF-14, an optional NRLMSISE-00 atmospheric density model,
and additional JPL ephemerides (DE441, DE442, and DE442s). The default
atmosphere and magnetic field remain available, while callers can select
these models for their simulation and orbital state. See the
`magnetic-field models <https://github.com/nscheuer/Generalized_ADCS/blob/main/ADCS/environment/magnetic_field/README.md>`__
and `atmosphere models <https://github.com/nscheuer/Generalized_ADCS/blob/main/ADCS/environment/atmosphere/README.md>`__
for configuration details.

Force and disturbance modeling
------------------------------

Disturbances now have a force interface as well as torque. The new force
stack includes aerodynamic drag, solar radiation pressure, body-fixed
thrust applied at a point, and a Sentman free-molecular drag model.
The work is tracked in pull requests
`#245 <https://github.com/nscheuer/Generalized_ADCS/pull/245>`__,
`#246 <https://github.com/nscheuer/Generalized_ADCS/pull/246>`__,
`#247 <https://github.com/nscheuer/Generalized_ADCS/pull/247>`__,
`#248 <https://github.com/nscheuer/Generalized_ADCS/pull/248>`__, and
`#249 <https://github.com/nscheuer/Generalized_ADCS/pull/249>`__.

Other features and fixes
------------------------

* `#233 <https://github.com/nscheuer/Generalized_ADCS/pull/233>`__: add
  ``DensityNoise``, which scales noise from a density by the sample time.
* `#234 <https://github.com/nscheuer/Generalized_ADCS/pull/234>`__: model
  as-built sensor scale factors and misalignment.
* `#235 <https://github.com/nscheuer/Generalized_ADCS/pull/235>`__ and
  `#251 <https://github.com/nscheuer/Generalized_ADCS/pull/251>`__: add
  ``BiasWithInstability`` with Gauss-Markov drift and correct its process
  spectral density term.
* `#257 <https://github.com/nscheuer/Generalized_ADCS/pull/257>`__,
  `#258 <https://github.com/nscheuer/Generalized_ADCS/pull/258>`__, and
  `#259 <https://github.com/nscheuer/Generalized_ADCS/pull/259>`__: expose
  Keplerian elements on ``Orbital_State``, fit SGP4 mean elements to the
  initial state, and make the conversion units explicit. Element attributes
  use radians.
* Refresh the benchmarks, ``simulate()`` examples, README, and package
  metadata for this release.
