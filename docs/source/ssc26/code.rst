:orphan:

.. _ssc26-code:

==================
SSC26 · The code
==================

Repository: https://github.com/nscheuer/Generalized_ADCS

.. code-block:: console

   git clone https://github.com/nscheuer/Generalized_ADCS.git
   cd Generalized_ADCS
   pip install .

Run the poster's code
=====================

Every snippet printed on the poster is executed verbatim by this script:

.. code-block:: console

   python papers/SSC26_poster/verify_snippets.py

It exits non-zero if any printed block stops working, so the poster and the
library cannot drift apart.

Where each stage lives
======================

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Stage
     - Module
   * - 1 · Goal formulation
     - ``ADCS/pipeline/goal_formulation/`` — ``attitude_error.py``,
       ``quat_set.py`` (reduced → full lift), ``omega_ref.py``,
       ``world_vectors.py``, ``conventions.py``
   * - 2 · Control law
     - ``ADCS/pipeline/control_law/`` — ``law_interface.py`` (the contract),
       ``pd_law.py``, ``sliding_mode_law.py``
   * - 4 · Compensation
     - ``ADCS/pipeline/compensation/`` — ``gyroscopic.py``,
       ``frame_rotation.py``, ``disturbance_ff.py``, ``damping_injection.py``
   * - 5 · Allocation
     - ``ADCS/pipeline/allocation/`` — ``lp.py``, ``qp.py``, ``qpw.py``,
       ``qpc.py``, ``pseudoinverse.py``, ``magnetic_cross.py``, ``momentum.py``
   * - Orchestration
     - ``ADCS/pipeline/pipeline_controller.py``, ``ADCS/pipeline/data.py``

``PipelineController`` subclasses the framework's ``Controller``, so it drops
into ``ADCS.simulate`` and ``ADCS.mc.monte_carlo_runner.MonteCarloRunner``
without changes to either.

Design documents
================

The block interfaces are specified before they are implemented:

- ``pipeline_spec.md`` — stage contracts and the ``LawInterface`` struct
- ``goal_formulation_spec.md`` — goal → error-signal routing
- ``allocation_spec.md`` — allocator formulations and momentum management

Tests
=====

.. code-block:: console

   pytest testing/test_pipeline/

Includes ``test_pipeline_vs_lovera.py``, which asserts the pipeline reproduces
the published ``MTQ_Lovera`` controller to machine precision across six states
including actuator saturation.

Reproducing published numbers
=============================

``pyproject.toml`` declares compatible version *ranges* so the package installs
into an existing environment. For deterministic campaign reruns use the exact
pins instead:

.. code-block:: console

   pip install -r requirements-repro.txt
   pip install -e . --no-deps
