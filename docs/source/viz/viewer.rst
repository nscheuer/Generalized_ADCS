3-D viewer
==========

The viewer (``viewer/``, Godot 4.7, GDScript) shows a simulation from a
spacecraft close-up out to the Earth–Moon system: Earth with day and night
imagery, clouds, terrain shading and ocean glint, the Moon, the Sun, catalogue
stars, and the team's spacecraft model. It reads only the
:doc:`visualization log <vizlog_v1>` and :doc:`stream <protocol_v1>`, plus the
:doc:`asset cache <assets>` and an :doc:`imported model <spacecraft_models>`.

Running it
----------

.. code-block:: bash

   python -m ADCS.viz.assets build                        # once: Earth, Moon and star assets
   python -m ADCS.viz view run.vizlog --viewer PATH/TO/Godot_v4.7.2-stable_win64.exe

.. code-block:: python

   from ADCS.viz import VizServer
   with VizServer() as server:                            # live: watch a running simulation
       print(server.address, server.token)                # or launch the viewer with them
       ADCS.simulate(..., viz_stream=server)

Until a packaged viewer is published, the "viewer" is a plain Godot 4.7.2
binary plus the repository's ``viewer/`` project, which ``view`` finds
automatically (or set ``ADCS_VIEWER_BIN`` / ``--viewer-project``). From WSL,
``view`` runs the Windows Godot natively on the Windows GPU, copies the
project to ``%LOCALAPPDATA%\generalized_adcs\viewer_project`` and serves data
over ``127.0.0.1``. The viewer can also open a log directly:
``Godot --path viewer -- --log run.vizlog --assets <cache>``.

Controls
--------

.. list-table::
   :widths: 25 75

   * - Mouse drag / wheel
     - orbit the target / zoom (logarithmic, ~10 cm to beyond the Moon)
   * - Space, ``,`` ``.``
     - play/pause, slower/faster (−1000× to 10 000×)
   * - 1 / 2 / 3
     - around the spacecraft in the LVLH / inertial / body frame
   * - 4
     - around Earth
   * - L, A, H, Home
     - follow live data, body axes, help, reset view

The timeline at the bottom scrubs through the run; the HUD shows UTC,
altitude, speed, eclipse state, pointing error and frame rate.

How it works
------------

* **Precision.** Positions stay in float64 until the camera position is
  subtracted; only camera-relative values reach the GPU. Godot's float32
  projection fails beyond far/near ≈ 10⁷, so the frustum follows the
  spacecraft (far = near × 10⁶) and Earth, Moon, Sun and stars are drawn in
  *scaled space*: shrunk about the camera to a log-compressed, strictly
  increasing distance (same image, correct depth order). See
  ``tools/viz_validation/PHASE0.md``.
* **Earth orientation** comes only from the log (``q_itrs_gcrs``); all Earth
  maps are cube maps sampled with the ITRS direction.
* **Spacecraft.** The model's levels of detail arrive as ``ASSET`` messages
  (or from the paths in ``scene.json`` for a log read from disk) and switch
  with on-screen size. Without a model, the drag/SRP panels are drawn. Wheels
  spin with the logged wheel speed. Sunlight on the spacecraft is scaled by
  the logged ``shadow_factor``.
* **Not yet:** atmosphere, cloud shadows and physical exposure (Phase 6);
  vector, cone and track overlays (Phase 7); video export (Phase 8). The Moon
  frame is approximated (near side toward Earth, libration ignored).

Checks
------

* ``viewer/tests/run_tests.gd``: headless unit tests (maths, frame store,
  clock, scaled space, protocol parsing, file source), run by
  ``testing/test_viz/test_viewer_godot.py`` when Godot is available.
* ``tools/viz_validation/run_viewer_checks.py``: end-to-end checks of the real
  viewer through ``python -m ADCS.viz view``'s code path, on a real
  closed-loop simulation with an imported model. Results in
  ``tools/viz_validation/VIEWER_CHECKS.md``.
