A Modular Open-Source ADCS Framework for Small Satellite Development and Testing
================================================================================

Patrick McKeen\ :sup:`1`, Niclas Bennet Darith Scheuer\ :sup:`2`,
Kerri Cahoy\ :sup:`1`

:sup:`1` Massachusetts Institute of Technology ·
:sup:`2` ETH Zürich

Poster, SmallSat Europe 2026, Amsterdam, Netherlands, May 28, 2026.

Read it
-------

- **Paper page** — https://smallsateurope.com/paper/a-modular-open-source-adcs-framework-for-small-satellite-development-and-testing/
- **Technical paper PDF** — https://smallsateurope.com/wp-admin/admin-ajax.php?action=jfp_download_paper&paper_id=721&attachment_id=3789
- **Contact the authors** — :doc:`../ssc26/contact`

Abstract
--------

Advanced attitude determination and control (ADCS) techniques are often
difficult for small satellite teams to adopt due to tightly coupled
implementations and the need to rapidly evaluate alternative architectures.
Evaluating prospective ADCS options may require significant modification of
existing code for specific actuator combinations, sensor hardware, orbits,
mission goals, and control laws. Existing options are limited: Basilisk and
OpenSatKit use C++ cores with Python wrappers and message-passing architectures
that are difficult to adapt, and most options focus on orbit mechanics or
overall flight software rather than closed-loop ADCS. This paper presents an
open-source, modular ADCS framework in pure Python that enables rapid
development, repeatable testing, and flight-oriented integration across
heterogeneous spacecraft.

The package supports torque- and command-level control, bound-respecting
allocation, and full- and reduced-attitude objectives. Estimation, control,
planning, and actuator/sensor modeling are implemented as interchangeable
modules reconfigured through configuration rather than code restructuring.
Integrated simulation includes attitude and orbit propagation, configurable
sensor and actuator models with bias and noise, and environmental disturbances.
An actuator-aware planner supports underactuated architectures, while
hardware-in-the-loop capability enables testing on mission-representative
computers.

We demonstrate the framework through a variety of case studies including a 3U
CubeSat with three magnetorquers and one reaction wheel, a magnetorquer-only 1U,
and a 6U with three reaction wheels and thrusters. Across these, we showcase
actuator and sensor failure modes, full- and reduced-attitude goals, and
momentum management. HIL testing on a Raspberry Pi is demonstrated in the same
package. Adapting a published control law to a new actuator configuration
requires less than 5 lines of configuration changes. The framework underpins
flight software for an Earth-observing CubeSat with visible and long-wave
infrared imagers, three magnetorquers, and one reaction wheel, without star
trackers or propulsion.

Presentation
------------

:download:`Open or download the SmallSat Europe presentation (PDF) <../_static/papers/package/smallsat_europe_kiosk_v4.pdf>`

.. raw:: html

   <div class="paper-pdf-preview">
     <object
       data="../_static/papers/package/smallsat_europe_kiosk_v4.pdf#page=1&amp;view=FitH"
       type="application/pdf"
       aria-label="Preview of the SmallSat Europe 2026 Generalized ADCS presentation">
       <p>Your browser cannot display the PDF preview.
         <a href="../_static/papers/package/smallsat_europe_kiosk_v4.pdf">Open the presentation</a>.
       </p>
     </object>
   </div>

Cite it
-------

.. code-block:: bibtex

   @inproceedings{mckeen2026modular,
     title     = {A Modular Open-Source ADCS Framework for Small Satellite Development and Testing},
     author    = {McKeen, Patrick and Scheuer, Niclas Bennet Darith and Cahoy, Kerri},
     booktitle = {SmallSat Europe 2026},
     year      = {2026},
     address   = {Amsterdam, Netherlands},
     url       = {https://smallsateurope.com/paper/a-modular-open-source-adcs-framework-for-small-satellite-development-and-testing/},
   }
