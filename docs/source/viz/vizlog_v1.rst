Visualization log format (adcs-vizlog v1)
==========================================

An *adcs-vizlog* is the hand-off between a simulation and a 3-D viewer. The
viewer reads only this format, never ADCS objects, so it can be written in any
language or engine. This page is the normative specification; the reference
implementation is :mod:`ADCS.viz` (:class:`~ADCS.viz.VizLogWriter`,
:class:`~ADCS.viz.VizLog`).

Producing a log
---------------

.. code-block:: python

   results = ADCS.simulate(..., viz_log="run.vizlog")          # single run
   results = simulate_mc(..., viz_log="mc.vizlog")              # one vizlog run per MC run
   ADCS.viz.results_to_vizlog(results, "run.vizlog")            # from results in memory

.. code-block:: bash

   python -m ADCS.viz convert saved.sim run.vizlog              # from a saved .sim file
   python -m ADCS.viz info run.vizlog

The log is written after the simulation finishes; the simulation loop is not
affected.

Directory layout
----------------

.. code-block:: text

   run.vizlog/
     manifest.json                 metadata, channel table, runs
     scene.json                    static spacecraft description
     runs/<run_id>/<channel>.f64   one file per channel
     runs/<run_id>/rows.json       {"n_rows": N, "complete": true|false}

* Channel files are **little-endian IEEE-754 float64**, row-major, shape
  ``(n_rows, width)``, with no header. Integer-valued channels (e.g.
  ``target_kind``) are stored as float64 too. Missing values are NaN.
* Every run contains every channel listed in the manifest.
* ``run_id`` matches ``[A-Za-z0-9_-]{1,64}``.
* JSON files are strict JSON (no NaN; non-finite numbers become ``null``).

Incomplete runs
~~~~~~~~~~~~~~~

Writers that append rows rewrite ``rows.json`` atomically after each flush.
If ``complete`` is false (a crash, or a run still being written), readers must
use ``min(file_size // (8 * width))`` over the run's channel files as the
number of rows, which drops any partially written row.

manifest.json
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``format``
     - ``"adcs-vizlog"``
   * - ``schema_version``
     - ``1``. Readers must reject versions they do not know.
   * - ``epoch``
     - ``{"jd_tt_whole", "jd_tt_frac", "iso_utc"}``: the instant ``t_s = 0``
       refers to, as a two-part TT Julian date (sum the parts) and, for
       people, ISO-8601 UTC.
   * - ``time_scale``
     - ``"TT"``
   * - ``frames``
     - ``GCRS`` (geocentric, ICRS-aligned axes), ``ITRS`` (Earth-fixed),
       ``BODY`` (spacecraft body frame).
   * - ``quaternion_convention``
     - Hamilton, scalar-first ``[q0, q1, q2, q3]``, active. ``q_A_B`` rotates
       vectors expressed in B into A: :math:`v_A = R(q_{A\_B})\,v_B`.
   * - ``channels``
     - List of ``{name, width, unit, frame, required, description,
       labels?, provenance?}``.
   * - ``runs``
     - List of ``{id, n_rows, dt_s, ...}``.
   * - ``generator``
     - Producing software: name, version, git SHA, Python version.
   * - ``ephemeris``
     - Kernel name, path, SHA-256, and Earth-orientation source/hash.
   * - ``earth``, ``bodies``, ``shadow_model``
     - Body radii and the shadow model used for ``shadow_factor``.

Required channels
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 8 10 10 52

   * - Name
     - Width
     - Unit
     - Frame
     - Meaning
   * - ``t_s``
     - 1
     - s
     - --
     - TT seconds since the manifest epoch
   * - ``r_sc``
     - 3
     - km
     - GCRS
     - spacecraft position
   * - ``v_sc``
     - 3
     - km/s
     - GCRS
     - spacecraft velocity (for Hermite interpolation)
   * - ``q_gcrs_body``
     - 4
     - --
     - --
     - attitude; consecutive rows are sign-continuous
   * - ``w_body``
     - 3
     - rad/s
     - BODY
     - angular velocity of BODY relative to GCRS
   * - ``r_sun``
     - 3
     - km
     - GCRS
     - apparent geocentric Sun position (light time and aberration applied)
   * - ``q_itrs_gcrs``
     - 4
     - --
     - --
     - Earth orientation, GCRS to ITRS (sign-continuous). Viewers must take
       Earth rotation from this channel rather than compute it.
   * - ``shadow_factor``
     - 1
     - --
     - --
     - visible fraction of the solar disk, conical umbra/penumbra model (0..1).
       Viewers scale sunlight on the spacecraft by this value.

Optional channels
-----------------

``r_moon`` (km, GCRS, apparent), ``sunlit_binary`` (the eclipse flag the
simulation itself used), ``b_field`` (T, GCRS), ``rho`` (kg/m³), ``h_wheel``
(N·m·s, wheel momentum relative to the body, one column per wheel),
``h_rw_body`` (N·m·s, BODY, total wheel momentum ``h_wheel @ axes``),
``wheel_speed`` (rad/s, ``h_wheel / J``), ``u_cmd`` (actuator commands, columns
labelled by actuator id), ``tau_act_body`` and ``tau_dist_body`` (N·m, BODY),
``target_kind`` / ``target`` / ``w_target`` / ``boresight_body`` /
``pointing_err`` (active goal; see below), ``q_est`` / ``w_est`` / ``r_est`` /
``v_est`` (estimates), and ``sensor_<id>`` (one channel per sensor block of the
simulation's reading vector, unit ``native``).

Two more families may be added by the caller
(``results_to_vizlog(..., extra_channels={...})``): ``joint_<node>`` (rad,
width 1) turns the hinged model node ``<node>`` about its ``axis_body``, and
``user_<name>`` (any width; give it 3 columns to draw it as an arrow) carries
anything else.

``target_kind``: 0 none, 1 direction (``target = [nan, unit vector GCRS]``),
2 attitude (``target = q_gcrs_body``), 3 position (``target = [nan, position
km GCRS]``).

Torque channels carry ``provenance``. ``nominal_recompute`` means they were
recomputed after the run with bias applied but noise disabled; with noise
enabled they are the nominal, not the exact integrated, torques.

scene.json
----------

Static, body-frame description of the spacecraft (lengths in metres):

* ``spacecraft``: ``mass_kg``, ``com_body_m``, ``inertia_ref_kgm2``,
  ``inertia_com_kgm2``.
* ``mesh``: ``null`` until a model is imported; the viewer then draws boxes
  from ``panels``. Otherwise ``{sidecar, sha256, coords, units, lods,
  nodes}``; ``nodes`` maps a model node to ``{role, actuator?, sensor?,
  articulation?: "spin"|"hinge", axis_body?, pivot_body?}``.
* ``panels``: surfaces used by drag/SRP, ``{area_m2, center_body_m,
  normal_body, eta_s, eta_d, eta_a, CD, used_by}``.
* ``actuators``: ``{id, index, type: "mtq"|"rw", axis_body, u_max, J_kgm2?,
  h_max_Nms?, node}``. ``index`` is the column in ``u_cmd``.
* ``sensors``: ``{id, type, slice: [i0, i1], boresight_body?, fov_half_rad?,
  fov_assumed?, bidirectional?, node}``. Fields of view are always half-cone
  angles in radians, whatever convention the sensor class uses.
* ``boresights``: named body-frame boresights; ``disturbances``: class names.
* ``timeline`` (optional): ``{bookmarks: [{id, t_s, label, note, color}],
  sites: [{name, lat_deg, lon_deg, alt_m, min_elev_deg, color, itrs_m,
  up_itrs}]}`` from :mod:`ADCS.viz.timeline`; ``itrs_m`` / ``up_itrs`` are the
  site's WGS84 position [m] and local vertical, so readers need no geodesy.
* ``overlays`` (optional): what the viewer draws over the scene, a list of
  descriptors written by :mod:`ADCS.viz.overlays` (each has ``type`` and a
  unique ``id``; see :doc:`viewer`). Readers ignore unknown types.

Validation
----------

``testing/test_viz`` checks the recomputed Sun and Moon directions and
distances, and the sub-solar longitude implied by ``q_itrs_gcrs``, against JPL
Horizons at 20 epochs from 1990 to 2026 (fixtures from
``tools/fetch_horizons_fixtures.py``): Sun within 1″, Moon within 2″, sub-solar
longitude within 15″ (the Horizons reference is topocentric).
