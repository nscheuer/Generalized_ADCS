07: SALTRO Planner
==================

This tutorial uses SALTRO on the same BeaverCube 1 pointing task as Tutorial 06.
The control problem remains extremely difficult: 3 MTQ-only actuation with aggressive
retargeting requirements.

Compared with the Plan-and-Track trajectory planner, SALTRO is typically much faster.
The tradeoff is that SALTRO does not assume the same level of perfect field knowledge,
so you should expect differences in tracking quality depending on scenario and tuning.

.. list-table:: Simulation Configuration
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Description
   * - **Satellite**
     - BeaverCube 1 generated from the built-in satellite factory.
   * - **Actuation**
     - 3 MTQ-only attitude control (underactuated, no reaction wheels).
   * - **Controller**
     - SALTRO planner/tracker with default SALTRO planner settings.
   * - **Goal Sequence**
     - Piecewise inertial goals with no-goal interval between retargeting segments.
   * - **Orbit**
     - Circular-like LEO test orbit initialized from position and velocity vectors.

.. code-block:: python

    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
    import ADCS as ADCS
    import numpy as np
    import matplotlib.pyplot as plt

    satellite = ADCS.satellite_factory.create_beavercube1_cubesat()

    x_0 = np.array([0, 0, 0] + [1, 0, 0, 0]) # w, q, h

    planner_settings = ADCS.controller.saltro.PlannerSettings(est_sat=satellite)
    controller = ADCS.controller.SALTRO(est_sat=satellite, planner_settings=planner_settings)

    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))
    goal_timeline = {0.0: ADCS.goals.ECI_Goal(np.array([1, 0, 0])), 400.0: ADCS.goals.No_Goal(), 700.0: ADCS.goals.ECI_Goal(np.array([0, 0, 1]))}
    goallist = ADCS.GoalList(goal_timeline=goal_timeline, time_units="seconds", start_juliantime=0.22)

    results = ADCS.simulate(
        x=x_0,
        satellite=satellite,
        controller=controller,
        goal=goallist,
        os0=os0,
        dt=1.0,
        tf=1000.0
    )

    ADCS.plot(
        results,
        ADCS.plots.AnimationPlot(),
        layout=(1,1),
        title="3+1 ALTRO Reduced",
    )

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "reference"]),
        layout=(1,1),
        title="3+0 ALTRO Mixed",
    )

    ADCS.plot(
        results,
        ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
        ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
        ADCS.plots.TargetHistogram(bin_width=5.0),
        ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
        layout=(2,2),
        title="3+0 ALTRO Mixed",
    )

    ADCS.plot(
        results,
        ADCS.plots.ControlPlotSingle(index=0, title="Magnetorquer 1", units="Am²"),
        ADCS.plots.ControlPlotSingle(index=1, title="Magnetorquer 2", units="Am²"),
        ADCS.plots.ControlPlotSingle(index=2, title="Magnetorquer 3", units="Am²"),
        layout=(3,1),
        title="3+0 ALTRO Mixed",
    )

    plt.show()

In practice, SALTRO provides a strong speed-accuracy tradeoff for hard magnetic-only
pointing tasks and is often preferred when fast replanning is required.

Simulation Results
------------------

Add your figures for attitude, control, and target-tracking performance here.
