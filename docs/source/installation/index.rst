Installation
============

Generalized ADCS needs **Python 3.10 or newer**. Pick your operating system,
then choose between installing from PyPI and cloning the repository.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * -
     - pip install
     - GitHub clone
   * - Core ADCS functionality and plotting
     - ✅
     - ✅
   * - Bundled examples (``python -m ADCS.examples...``)
     - ✅
     - ✅
   * - Full examples, debug scripts, and tutorials
     -
     - ✅
   * - Offline documentation
     -
     - ✅
   * - Optional C++ add-ons (built separately)
     -
     - ✅

.. tab-set::
   :sync-group: os

   .. tab-item:: Linux / WSL
      :sync: linux

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: pip install

            Lightweight setup.

            .. code-block:: bash

               python3 -m venv adcs-venv
               source adcs-venv/bin/activate
               pip install generalized-adcs
               python -m ADCS.examples.basic_simulation

         .. grid-item-card:: GitHub Clone

            Full examples, debug scripts, tutorials and offline documentation.

            .. code-block:: bash

               git clone https://github.com/nscheuer/Generalized_ADCS.git
               cd Generalized_ADCS
               python3 -m venv .venv
               .venv/bin/python -m pip install -e .
               .venv/bin/python -m ADCS.examples.basic_simulation

   .. tab-item:: Windows
      :sync: windows

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: pip install

            Lightweight setup.

            .. code-block:: powershell

               python -m venv adcs-venv
               .\adcs-venv\Scripts\activate
               pip install generalized-adcs
               python -m ADCS.examples.basic_simulation

         .. grid-item-card:: GitHub Clone

            Full examples, debug scripts, tutorials and offline documentation.

            .. code-block:: powershell

               git clone https://github.com/nscheuer/Generalized_ADCS.git
               cd Generalized_ADCS
               python -m venv .venv
               .\.venv\Scripts\python.exe -m pip install -e .
               .\.venv\Scripts\python.exe -m ADCS.examples.basic_simulation

   .. tab-item:: macOS
      :sync: macos

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: pip install

            Lightweight setup.

            .. code-block:: bash

               python3 -m venv adcs-venv
               source adcs-venv/bin/activate
               pip install generalized-adcs
               python -m ADCS.examples.basic_simulation

         .. grid-item-card:: GitHub Clone

            Full examples, debug scripts, tutorials and offline documentation.

            .. code-block:: bash

               git clone https://github.com/nscheuer/Generalized_ADCS.git
               cd Generalized_ADCS
               python3 -m venv .venv
               .venv/bin/python -m pip install -e .
               .venv/bin/python -m ADCS.examples.basic_simulation

The source checkout needs Git. The PyPI path runs the same included example
without downloading any additional files.

Check that it worked
--------------------

The last command of either path runs a 30-second free-rotation simulation.
After the progress bars, it prints:

.. code-block:: text

   Final angular velocity [rad/s]: [0.         0.         0.01745329]
   Final attitude quaternion [scalar, x, y, z]: [0.96814764 0.         0.         0.25038   ]

Next, follow the :doc:`../getting_started/quickstart`, or see
:doc:`../getting_started/examples` to get the full example collection.

Optional extras
---------------

Extras add optional dependencies. Install one with
``pip install "generalized-adcs[<extra>]"``, or ``pip install -e ".[<extra>]"``
from a clone.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - What it adds
   * - ``viz``
     - 3-D orbit animation: PyVista orbit plots (``ADCS.plots.AnimationPlot``),
       Godot asset texture processing, and Pillow-based rendering and export
   * - ``viz-video``
     - MP4 export from the Godot viewer when no ``ffmpeg`` is on ``PATH``
   * - ``viz-mesh``
     - Spacecraft model import (glTF/STL) with level-of-detail decimation
   * - ``viz-assets``
     - Rebuilding the viewer's texture assets
   * - ``nrlmsise``
     - The NRLMSISE-00 atmosphere model (via ``pymsis``)
   * - ``ppigrf``
     - The legacy geomagnetic field model, selectable with
       ``magnetic_model="igrf_ppigrf"`` to reproduce results from 0.1.8 and
       earlier
   * - ``dev``
     - Test and lint tooling
   * - ``docs``
     - Documentation build
   * - ``cpp``
     - Build toolchain for the optional C++ add-ons (source checkout only; see
       below)

For bit-reproducible paper campaigns, install the exact pins from a clone
instead: ``pip install -r requirements-repro.txt``.

Optional C++ add-ons
--------------------

Optional C++ add-ons can be installed after the core framework. Their source
code lives in the ``OldPlanner`` and ``SALTRO`` Git submodules, so they need a
clone of the repository. Neither is needed for the pure-Python framework.

.. toctree::
   :maxdepth: 2
   :caption: Add-Ons

   trajectory_planner_addon
   saltro_addon

See :doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for an overview of
trajectory optimization add-ons and where to start.
