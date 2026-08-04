:orphan:

.. _ssc26-contact:

===============
SSC26 · Contact
===============

Thanks for scanning. We would genuinely like to hear which control law you
want to drop in.

Get in touch
============

.. TODO(ssc26): if a hosted form is preferred over mailto, put its URL here
   and drop the mailto block. A Sphinx page is static, so the form itself has
   to be hosted elsewhere (Google Forms, Formspree, ...).

- **Email** — patrick@learn.ventures
- **Issues and feature requests** —
  https://github.com/nscheuer/Generalized_ADCS/issues
- **Discussions** —
  https://github.com/nscheuer/Generalized_ADCS/discussions

Useful things to tell us
========================

If you are asking about adapting a specific control law, these four details
let us answer concretely:

1. The law — a citation, or the torque expression
2. What error signals it consumes (attitude representation, rate or no rate)
3. Which compensation terms it already performs internally
4. Your actuator set (magnetorquers, wheels, thrusters; how many, what axes)

That maps directly onto the ``LawInterface`` declaration in
:doc:`the code <code>`, which is the whole adaptation problem in one struct.

Contributing
============

See :doc:`../contributing/index`. The library is MIT-licensed.
