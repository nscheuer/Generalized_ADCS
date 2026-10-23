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
   * - O, A
     - all overlays on/off, axis triads on/off
   * - ← →, Shift+← →
     - previous / next sample, previous / next event or bookmark
   * - B, T, F, I
     - bookmark, type a time, fit the timeline, state card
   * - L, H, Home
     - follow live data, help, reset view

The HUD (top left) shows altitude, speed, eclipse state, pointing error and
frame rate; the timeline (below) shows the time.

Timeline
--------

.. code-block:: text

   ▶ ⏮ ⏭ [60×]   2026-03-14 12:41:07.2 UTC  T+00:41:07 · sample 1234     ⧉ ★ ☰ ↔
   Sun     ░░░░sunlit░░░░▓▓▓▓umbra▓▓▓▓░░░░░░░    ◆ bookmarks on the ruler
   Boston          ▬▬▬▬            ▬▬▬            one lane per ground site
           [=========window=========]             the whole run

* **Find a moment.** Drag on the track to scrub (playback pauses while you
  drag); the wheel zooms about the cursor, from the whole run down to 1 s;
  Shift+wheel or a right/middle drag pans; double-click or F fits the run.
  Hovering shows the UTC time, altitude, light and passes at that point.
* **Land on it exactly.** Scrubbing snaps to logged samples when they are a few
  pixels apart, and the readout says ``sample k`` or ``interpolated``. Left /
  Right step one sample; Shift+Left / Right jump to the previous / next eclipse
  change, pass rise or set, or bookmark. Click the time (or press T) to type
  one: a UTC date-time (``2026-03-14 12:41:07``), a UTC clock time on the date
  shown (``12:41``), or elapsed time (``T+41:07``, ``+2460``, ``T+1h2m``).
* **See the attitude.** While paused, the state card shows roll / pitch / yaw
  relative to LVLH (3-2-1 sequence; LVLH: Z nadir, Y negative orbit normal),
  the quaternion, body rates, pointing error, altitude, light and passes in
  progress (I hides it; Copy puts it on the clipboard as JSON).
* **Come back to it.** B bookmarks the playhead and lets you name it; the list
  (☰) jumps to, renames or deletes bookmarks and copies them all, with the
  state at each, as JSON. Your bookmarks are saved per run (in the viewer's
  user data folder) and return when the same run is opened again; bookmarks
  from the log are shown locked.

Lanes come from the log: light from ``shadow_factor`` (sunlit ≥ 0.999,
umbra ≤ 0.001, penumbra between), passes from ``r_sc`` and ``q_itrs_gcrs`` for
each ground site in ``scene.json``, by the same rule as
:func:`ADCS.viz.timeline.ground_passes` (geodetic elevation above the site's
``min_elev_deg``, rise and set interpolated between rows). They also fill in
during a live stream.

.. code-block:: python

   from ADCS.viz import timeline as tl
   results_to_vizlog(results, "run.vizlog",
       sites=[tl.GroundSite("Boston", 42.36, -71.06, min_elev_deg=10)],
       bookmarks=[tl.Bookmark(t_s=840, label="detumble done"),
                  tl.Bookmark(utc="2026-03-14T12:41:07Z", label="first contact")])
   tl.ground_passes(VizLog.open("run.vizlog"), tl.GroundSite("Boston", 42.36, -71.06))
   # -> [Pass(t_rise, t_set, t_max, max_elev_deg, partial), ...]

:func:`~ADCS.simulate` adds a site for every ``Coordinate_Goal`` target.

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
* **Atmosphere** (Hillaire 2020): tables from the asset pipeline, ray-marched
  in the shaders. Sunlight reaching the ground is attenuated (reddened
  terminator); the air between camera and ground is applied per vertex over
  most of the disk and per pixel near the limb; a transparent shell draws the
  limb glow. Clouds cast shadows. Sunlight on the spacecraft is coloured by
  the atmosphere when the Sun is seen through the limb; earthshine lights it
  from the nadir.
* **Moon**: Lommel–Seeliger reflectance (no limb darkening at full Moon). Its
  frame is approximated (near side toward Earth, libration ignored).
* **Exposure** (E, [ ]): *artistic* (default) compresses stars and the Sun so
  they show next to a lit Earth; *physical* keeps true relative brightness:
  a star's flux lands in one pixel (π E / (Ω_pixel E_sun)), so stars vanish
  beside a sunlit Earth and appear at long exposures, as for a real camera.
* **Quality** (Q, ``--quality``): ``high`` (MSAA 4×), ``balanced`` (FXAA,
  default), ``fast`` (no glow, fewer atmosphere steps). Measured on an Intel
  Arc 140V (integrated) with atmosphere, clouds and a model, GPU p50/p99:
  balanced 1080p 5.1/10.4 ms, 1440p 6.9/19.0 ms.
* **Articulation.** Model nodes marked ``"articulation": "hinge"`` in the
  model config turn about ``axis_body`` through ``pivot_body`` by the logged
  ``joint_<node>`` angle (0 = as modelled); wheels spin from ``wheel_speed``.
* **Not yet:** video export (Phase 8).

Overlays
--------

What is drawn over the scene is decided in Python and stored in the log's
``scene.json``, so a recorded log and a live stream show the same thing.
:func:`~ADCS.viz.results_to_vizlog` and live streams add
:func:`~ADCS.viz.overlays.default_overlays`: body axes, control and
disturbance torque, wheel momentum (``h_rw_body``), magnetic field and Sun
direction, sensor fields of view (not for assumed ones), orbit and ground
tracks, and the pointing target. Choose your own with ``overlays=[...]``:

.. code-block:: python

   from ADCS.viz import overlays as ov
   results_to_vizlog(results, "run.vizlog", overlays=[
       ov.Triad("lvlh"),
       ov.Arrow("tau_act_body", label="control torque"),               # log scale
       ov.Arrow("user_thrust", scale="physical", units_per_m=0.01),     # 0.01 N per metre
       ov.Arrow("@sun", label="Sun", scale="fixed_px", px=100),
       ov.SensorCone("startracker0"),
       ov.OrbitTrack(past_s=5400, future_s=1800), ov.GroundTrack(),
   ], extra_channels={"user_thrust": thrust_body})                    # (n_rows, 3) array

.. list-table::
   :widths: 22 78

   * - ``Arrow``
     - from the centre of mass. ``source``: a 3-wide channel (BODY or GCRS,
       from the manifest) or ``@sun``, ``@nadir``, ``@velocity``,
       ``@boresight:<name>``. ``scale``: ``log`` (``px_min`` at ``ref``, the
       10th percentile of the log by default, +``px_per_decade`` per factor
       10, at most ``px_max``), ``fixed_px`` (direction only) or ``physical``
       (``units_per_m``). Vectors that are not available (NaN) are hidden.
   * - ``Triad``
     - X/Y/Z axes of the ``body``, ``lvlh`` (Z nadir, Y −orbit normal) or
       ``gcrs`` frame.
   * - ``SensorCone``
     - half-angle and boresight from the scene's sensor entry.
   * - ``OrbitTrack``, ``GroundTrack``
     - the logged path from ``past_s`` before to ``future_s`` after now;
       the ground track is fixed on the rotating Earth, 3 km above the
       surface. One-pixel lines, hidden behind the Earth.
   * - ``TargetLine``
     - toward the active target (``target_kind`` / ``target``).
   * - ``Label``
     - text next to the spacecraft.

Spacecraft-attached overlays keep a constant size on screen whatever the
zoom, labels are 15 px high and pushed apart when they overlap, and arrows are
drawn over the model (``on_top``). The ``overlays`` check in
``run_viewer_checks.py`` compares rendered arrows and tracks with a float64
projection of the logged data, and ``perf_overlays`` measures their CPU cost
(about 0.26 ms per frame for the default set and 1 ms for 50 labelled arrows,
GDScript).

Checks
------

* ``viewer/tests/run_tests.gd``: headless unit tests (maths, frame store,
  clock, scaled space, protocol parsing, file source), run by
  ``testing/test_viz/test_viewer_godot.py`` when Godot is available.
* ``tools/viz_validation/run_viewer_checks.py``: end-to-end checks of the real
  viewer through ``python -m ADCS.viz view``'s code path, on a real
  closed-loop simulation with an imported model. Results in
  ``tools/viz_validation/VIEWER_CHECKS.md``.
* ``timeline`` / ``timeline_perf`` (in ``run_viewer_checks.py``): mouse and
  key events sent through the real GUI land on the time under the cursor and
  the playhead is drawn there; lanes equal the Python reference; the state
  card equals an independent roll / pitch / yaw; bookmarks survive a restart;
  drawing cost on a 24 h log.
* ``tools/viz_validation/run_epic_check.py``: renders the Earth from DSCOVR's
  position at the time of a real EPIC photograph and compares disk geometry,
  orientation (surface pattern vs time) and cloud-free colour. Results in
  ``tools/viz_validation/EPIC_CHECK.md``.
