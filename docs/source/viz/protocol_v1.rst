Visualization stream protocol (v1)
===================================

How a 3-D viewer receives data from Python, either a recorded
:doc:`vizlog <vizlog_v1>` (playback) or a simulation while it runs (live). The
reference implementation is :mod:`ADCS.viz.protocol`, with
:class:`~ADCS.viz.VizServer` (Python side) and :class:`~ADCS.viz.VizClient`
(a reference viewer, used by the tests).

Usage
-----

.. code-block:: bash

   python -m ADCS.viz view run.vizlog                 # serve a log and launch the viewer (see viewer page)
   python -m ADCS.viz view run.vizlog --no-launch     # serve only; print args for a viewer started by hand
   python -m ADCS.viz listen 127.0.0.1:PORT --token T # reference client: print what a server sends

.. code-block:: python

   from ADCS.viz import VizServer

   with VizServer(mode="broadcast") as server:        # or mode="lockstep"
       print(server.address, server.token)            # give these to the viewer
       results = ADCS.simulate(..., viz_stream=server)

Roles and connection
--------------------

* Python **listens**; the viewer **connects** (TCP). This works the same on one
  machine and for a Windows viewer talking to a simulation in WSL2, where
  Windows forwards ``127.0.0.1`` into WSL (verified), so the server never needs
  to listen beyond the loopback interface.
* The launcher starts the viewer as::

     <viewer> -- --connect HOST:PORT --token TOKEN

  Godot hands arguments after ``--`` to the project (``OS.get_cmdline_user_args()``).
* The viewer's first message is ``CONNECT`` with the token. A wrong token or
  protocol version is answered with ``ERROR`` and the connection is closed.

Framing
-------

All integers are little-endian. Each message is a 20-byte header and a payload:

.. list-table::
   :header-rows: 1
   :widths: 10 10 80

   * - Offset
     - Size
     - Field
   * - 0
     - 4
     - magic ``ADVZ``
   * - 4
     - 2
     - protocol major version, ``uint16`` = 1
   * - 6
     - 2
     - message type, ``uint16``
   * - 8
     - 4
     - payload length in bytes, ``uint32`` (at most 64 MiB)
   * - 12
     - 8
     - sequence number, ``uint64``

Messages
--------

.. list-table::
   :header-rows: 1
   :widths: 6 12 14 68

   * - Type
     - Name
     - Direction
     - Payload
   * - 1
     - HELLO
     - server → viewer
     - JSON ``{"protocol": 1, "mode", "manifest", "scene"}``. ``mode`` is
       ``playback``, ``live-broadcast`` or ``live-lockstep``. ``manifest`` and
       ``scene`` are exactly a vizlog's ``manifest.json`` and ``scene.json``;
       the manifest's channel list fixes the row layout.
   * - 2
     - BLOCK
     - server → viewer
     - ``u16 n, run_id[n] (UTF-8), u32 row_start, u32 n_rows``, then each
       channel in manifest order as ``float64[n_rows][width]``: a slice of each
       vizlog channel file, concatenated.
   * - 3
     - ROW
     - server → viewer
     - ``u16 n, run_id[n]``, then ``float64`` values of every channel in
       manifest order: byte-identical to one row of the vizlog files.
   * - 4
     - END
     - server → viewer
     - JSON ``{"run_id": id}`` (a run is complete) or ``{}`` (session over).
   * - 5
     - ACK
     - viewer → server
     - empty; the header's sequence number is that of the acknowledged ROW
       (lockstep only).
   * - 6
     - CONNECT
     - viewer → server
     - JSON ``{"token", "client", "protocol": 1}``.
   * - 7
     - ERROR
     - either
     - JSON ``{"message"}``; the sender then closes the connection.
   * - 8
     - ASSET
     - server → viewer
     - ``u16 n, name[n] (UTF-8), u32 m, meta[m] (JSON), data``: a file the
       scene refers to, sent right after ``HELLO``. Used for the spacecraft
       model's levels of detail (``meta = {"kind": "model_lod", "lod", "sha256",
       "triangles"}``), so a viewer never needs the Python side's file paths
       (e.g. a Windows viewer and a simulation inside WSL).

Sessions
--------

**Playback.** ``HELLO``, ``ASSET`` messages (if the scene has a model), then for each run its ``BLOCK`` messages (rows in
order) and ``END {"run_id"}``, then ``END {}``. The connection stays open until
the viewer closes it; every viewer that connects receives the whole log, so a
viewer can hold all rows in memory and scrub freely.

**Live, broadcast** (default). ``HELLO``, ``ASSET`` messages, then ``ROW`` messages as the
simulation produces them, ``END {"run_id"}`` and ``END {}``. The simulation
never waits for the viewer: rows wait in a bounded queue (default 4096) and the
oldest are dropped if the viewer is slow or absent. Rows produced before the
viewer connects are delivered on connection, up to the queue size.

**Live, lockstep.** As broadcast, but each ``ROW`` is sent synchronously and
the simulation waits for its ``ACK`` (default timeout 10 s). Use it for
deterministic frame capture. If the viewer disconnects or times out, streaming
stops with a warning and the simulation continues. The simulation waits up to
30 s for a viewer before the first step.

Only one viewer is served per live session; playback serves any number, one
after another or at once.

Live rows contain the required channels plus ``r_moon``, ``sunlit_binary``,
``h_wheel``, ``wheel_speed`` and ``u_cmd``. Torques, goals, estimates and
sensor channels are only in the post-run vizlog (``simulate(..., viz_log=...)``).

Cost
----

Measured on the development machine (WSL2), 1500-step simulation, 3 reaction
wheels, ~5 ms per step without streaming: broadcast with no viewer connected
adds no measurable time; with a viewer, the extra time is dominated by the
viewer itself when it runs in the same Python process (as in the tests), about
0.5–1 ms per step.

Viewer binary
-------------

The viewer is a separate program, never part of the Python wheel. It is found
in this order: an explicit path, ``ADCS_VIEWER_BIN``, then a release installed
with ``python -m ADCS.viz install-viewer`` (pinned by version and SHA-256 in
``ADCS/viz/_viewer_manifest.json``; archives are checked and unpacked safely).
Inside WSL the Windows build is selected and installed under
``%LOCALAPPDATA%``. No release is published yet; until then run the viewer from
the Godot editor against ``view --no-launch``.
