Importing a spacecraft model
============================

Teams can show their own spacecraft in the 3-D viewer. The importer
(:mod:`ADCS.viz.mesh_import`) converts the model to the spacecraft body frame
in metres, checks it against the simulation's
:class:`~ADCS.satellite_hardware.satellite.Satellite`, and writes reduced
levels of detail plus a sidecar file that the visualization log points to.

.. code-block:: bash

   pip install "generalized-adcs[viz-mesh]"          # trimesh, fast-simplification
   python -m ADCS.viz.mesh_import spacecraft_viz.json --out model/ --satellite mysat.factory:make
   python -m ADCS.viz convert run.sim run.vizlog --model model/mysat.viz.json

.. code-block:: python

   from ADCS.viz.mesh_import import import_spacecraft
   res = import_spacecraft("spacecraft_viz.json", "model/", satellite=sat)
   print(res.warnings, res.errors)
   ADCS.viz.results_to_vizlog(results, "run.vizlog", model=res.sidecar)

Exit codes: 0 imported, 1 imported with warnings (only with ``--strict``),
2 not imported (errors).

Model formats
-------------

* **glTF 2.0 (.glb / .gltf), preferred.** Units are metres. Keeps named nodes,
  which is how wheels, panels and sensors are identified, and PBR materials.
  Export from Blender, Onshape, Fusion, SolidWorks (via plug-in) and others.
  Mesh compression (Draco, meshopt, quantization) is not supported; export
  without it.
* **STL / OBJ / PLY, accepted.** One static mesh with **no units**: ``"units"``
  is required.

Configuration
-------------

.. code-block:: json

   {
     "name": "mysat",
     "mesh": "mysat.glb",
     "units": "m",
     "cad_to_body": {"rotation": "gltf_y_up", "translation_m": [0.0, 0.0, -0.05]},
     "mass_kg": 2.6,
     "nodes": {
       "bus":     {"role": "body"},
       "wheel_x": {"role": "rw", "actuator": "rw0"},
       "array_1": {"role": "panel", "articulation": "hinge", "axis_body": [0, 1, 0],
                   "pivot_body": [0.05, 0.0, 0.1]},
       "tracker": {"role": "sensor", "sensor": "startracker0", "boresight_local": [0, 0, 1]}
     },
     "lods": [1.0, 0.25, 0.05],
     "max_triangles": 500000
   }

* ``mesh`` is relative to the configuration file.
* ``cad_to_body`` maps model coordinates (after unit scaling) to the body
  frame: a 4×4 matrix, or a rotation (3×3 or a preset) plus a translation in
  metres. Presets: ``identity``, and ``gltf_y_up``, which undoes the Z-up → Y-up
  conversion that Blender and most CAD tools apply when writing glTF.
* ``mass_kg`` defaults to the satellite's mass; ``density_kg_m3`` can be given
  instead when there is no satellite.
* A ``hinge`` node turns about ``axis_body`` through ``pivot_body`` (BODY
  metres; default: the corner of the part nearest the body origin) by the
  vizlog channel ``joint_<node>`` in radians, 0 being the pose in the file:
  ``results_to_vizlog(..., extra_channels={"joint_array_1": angles})``.
  Wheel nodes spin about their actuator's axis automatically.
* Node ids for ``actuator`` and ``sensor`` are the ones in the vizlog
  ``scene.json``: ``mtq0``, ``rw0``, ``startracker0``, ``sunsensor1``, … in the
  satellite's order.

Checks
------

Errors stop the import; warnings are reported in the sidecar and on the
console. Disagreements with the simulation are warnings, because CAD usually
leaves out internal mass and small parts.

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Check
     - Outcome
     - Rule
   * - Missing units (STL), unknown units or preset
     - error
     -
   * - Compressed glTF, non-uniform node scale, shear, mirroring
     - error
     - re-export without them
   * - Config names a node not in the model, or an unknown actuator/sensor id
     - error
     -
   * - Units look wrong
     - warning
     - model size vs. the size implied by the satellite's inertia
       (√(2 tr J / m)) differs by more than 10×; suggests the likely unit
   * - Open mesh (holes, inconsistent normals)
     - warning
     - normals are repaired where possible; open parts are drawn but left out of
       mass properties
   * - Centre of mass
     - warning
     - offset > max(1 cm, 2% of the model diagonal)
   * - Inertia
     - warning
     - relative Frobenius difference > 10%, assuming uniform density over the
       closed parts
   * - Principal axes
     - warning
     - > 5° (only for axes whose moments differ by ≥ 5%)
   * - Reaction wheel axis
     - warning
     - the node's symmetry axis (most distinct principal axis) > 2° from the
       actuator axis
   * - Sensor boresight
     - warning
     - the node's ``boresight_local`` (default +Z) > 1° from the sensor's boresight
   * - Drag/SRP panels
     - warning
     - mesh area facing each panel's normal within 10° and within
       max(1 cm, 2% diagonal) of its plane differs from the panel area by > 10%
   * - Triangle budget
     - warning
     - more than ``max_triangles`` (default 500k)

Output
------

* ``<name>_lod<i>.glb``: one file per level of detail, node names preserved,
  vertices in **BODY-frame metres as stored** (no glTF Y-up convention applied;
  the viewer applies the BODY → engine basis). Reduced levels use
  fast-simplification and replace textures with the material's average colour,
  because simplification drops texture coordinates.
* ``<name>.viz.json``: source file and hash, units, ``cad_to_body``, levels of
  detail with triangle counts and hashes, per-node role / actuator / sensor /
  articulation and check results, computed and declared mass properties, panel
  comparison, and the full report.

With ``model=`` (or ``convert --model``) the vizlog ``scene.json`` gets a
``mesh`` entry pointing at the sidecar and its levels of detail, and each
actuator and sensor gets the name of its model node. Without a model, the
viewer draws the drag/SRP panels.
