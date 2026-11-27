Release Notes
=============

What changed in each release of Generalized ADCS, newest first. To update an
existing install, run ``pip install --upgrade generalized-adcs``.

Latest releases
---------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 0.2.2 · Visualization and Environment
      :img-top: ../_static/release_notes/0_2_2_animation_preview.png
      :img-alt: ICESat-2 maneuvering above Earth at sunset
      :link: 0_2_2_visualization_and_environment
      :link-type: doc

      Godot animations, additional environmental models, and a new disturbance
      force stack.

   .. grid-item-card:: 0.2.1 · Attitude Determination and Performance
      :img-top: ../_static/release_notes/0_2_1_filter_tradeoff.png
      :img-alt: EKF, MEKF, UKF, and SRUKF simulation time per step versus attitude error at 100 seconds
      :link: 0_2_1_attitude_determination_and_speed
      :link-type: doc

      MEKF warm starts for large attitude errors, a 100x+ faster default
      geomagnetic field model, and robust estimator and simulation updates.

   .. grid-item-card:: 0.2.0 · New Estimation Framework
      :img-top: ../_static/release_notes/0_2_0_advanced_MEKF.png
      :img-alt: Advanced MEKF estimation results
      :link: 0_2_0_estimation_framework
      :link-type: doc

      Named state and covariance classes, a canonical measurement stack, shared
      process models and noise, a new attitude-estimator base, and eight new
      attitude estimators.

All releases
------------

.. toctree::
  :maxdepth: 1
  :caption: Release Notes

  0_2_2_visualization_and_environment
  0_2_1_attitude_determination_and_speed
  0_2_0_estimation_framework
  0_1_8_installable
  0_1_7_benchmark
  0_1_6_remote_execution
  0_1_5_bugfixes_and_optimizations
  0_1_4_numba_optimizations_2
  0_1_3_hardware_in_the_loop
  0_1_2_numba_optimizations
  0_1_1_earth_horizon
  0_1_0_saltro
  0_0_2_multi_boresight
  0_0_1_mc
  0_0_0_initial
