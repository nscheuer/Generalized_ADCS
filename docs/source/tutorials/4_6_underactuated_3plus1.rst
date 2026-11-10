Underactuated Control: 3+1
==========================

.. image:: ../_static/tutorials/controlling_spacecraft/underactuated_3plus1.gif
   :alt: MiRaTa tracks a daylight ground target in a fixed spacecraft view.
   :width: 100%

Three magnetorquers and one reaction wheel track a daylight ground target
at 16° S, 160° W, offset from nadir. The initial pointing error is about
30°. Yellow marks the body boresight; green marks the target direction.
This 3+1 architecture is discussed in [1]_.

Simulation and animation
------------------------

.. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/underactuated_3plus1.py
   :language: python
   :caption: underactuated_3plus1.py

Add plots
---------

.. container:: dynamics-visualization-grid

   .. container:: installation-card

      .. rubric:: Pointing error

      .. image:: ../_static/tutorials/controlling_spacecraft/underactuated_3plus1_angle_error.png
         :alt: Pointing error falls below 20 degrees early and converges near zero.
         :width: 100%

      .. code-block:: python

         import matplotlib.pyplot as plt

         ADCS.plot(
             results,
             ADCS.plots.TargetPlot(
                 modes=["real_target"], title="Ground-target pointing error"
             ),
             layout=(1, 1), figsize=(8, 4),
         )
         plt.show()

   .. container:: installation-card

      .. rubric:: MTQ and reaction-wheel commands

      .. image:: ../_static/tutorials/controlling_spacecraft/underactuated_3plus1_controls.png
         :alt: Magnetorquer and reaction-wheel control torques during the simulation.
         :width: 100%

      .. code-block:: python

         import matplotlib.pyplot as plt

         ADCS.plot(
             results,
             ADCS.plots.ControlPlotCombined(),
             ADCS.plots.ControlPlotSingle(
                 index=0, title="Reaction-wheel control torque"
             ),
             layout=(2, 1),
         )
         plt.show()

The single wheel cannot unload momentum while preserving pointing, so the
example sets ``c_gain=0``.

.. [1] N. Scheuer, P. McKeen, and K. Cahoy, “One Wheel Is Enough: Control
   Strategies for the 3+1 Actuator Architecture Across Small Satellite
   Mission Profiles”.
