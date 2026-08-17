:orphan:

.. _ssc26-run:

.. include:: _analytics.txt

===============================
SSC26 · Run it in your browser
===============================

No install, no local Python. Two cells take a 3-magnetorquer + 1-reaction-wheel
CubeSat from 90° off target to under a tenth of a degree.

Open the notebook
=================

.. TODO(ssc26): the Colab link needs the notebook on main, which arrives with
   the pipeline PR. This page is the QR target so the link can be corrected
   without reprinting the poster. Until then the install instructions below
   work today.

.. raw:: html

   <p>
     <a href="https://colab.research.google.com/github/nscheuer/Generalized_ADCS/blob/main/papers/SSC26_poster/SSC26_quickstart.ipynb">
       <img src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open in Colab">
     </a>
   </p>

If the badge above 404s, the notebook has not reached ``main`` yet — install
locally instead, it is one line:

.. code-block:: console

   pip install generalized-adcs

What it does
============

.. code-block:: python

   # the bus: 3 magnetorquers, 1 reaction wheel
   acts  = [ADCS.MTQ(axis=a, max_torque=0.2) for a in np.eye(3)]
   acts += [ADCS.RW(axis=np.array([0, 0, 1.0]), max_torque=0.0023,
                    J=5.7e-6, h=0.0, h_max=0.0036)]

   # point the +z boresight at an inertial direction, from 90 deg away
   res = ADCS.simulate(x=x0, satellite=sat, controller=ctrl,
                       goal=goal, os0=os0, dt=1.0, tf=3000.0)

Measured result: **90.2° → 0.056°** in 3000 s.

A note on Colab
===============

Colab ships its own NumPy, and ``numba`` constrains the version it works with,
so the install may downgrade NumPy and ask you to restart the runtime. If it
does, restart and re-run the second cell — the simulation itself takes a few
seconds.

Also on the poster
==================

- :doc:`index` — what the adapter is, and the code
- :doc:`paper` — the paper and how to cite it
- :doc:`code` — the repository
