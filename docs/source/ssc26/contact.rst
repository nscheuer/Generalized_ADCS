:orphan:

.. _ssc26-contact:

===============
SSC26 · Contact
===============

.. include:: _analytics.txt

Thanks for scanning. We would genuinely like to hear which control law you
want to drop in — tell us and we will help you port it.

Tell us about your law
======================

.. raw:: html

   <form class="ssc26-form" method="POST"
         action="https://formspree.io/f/mkjwqeqy">
     <input type="hidden" name="_subject"
            value="SSC26 poster — control law porting request">
     <p style="display:none">
       <label>Leave this empty<input name="_gotcha" tabindex="-1"></label>
     </p>

     <label for="ssc26-email">Your email</label>
     <input id="ssc26-email" type="email" name="email" required
            placeholder="you@university.edu">

     <label for="ssc26-law">Which control law?</label>
     <input id="ssc26-law" type="text" name="law"
            placeholder="a citation, or the torque expression">

     <label for="ssc26-signals">What does it consume?</label>
     <input id="ssc26-signals" type="text" name="signals"
            placeholder="quaternion error, MRP, rate / no rate …">

     <label for="ssc26-handles">What does it already do internally?</label>
     <input id="ssc26-handles" type="text" name="handles"
            placeholder="gyroscopic term, frame rotation, disturbance FF …">

     <label for="ssc26-hardware">Your actuator set</label>
     <input id="ssc26-hardware" type="text" name="hardware"
            placeholder="3 magnetorquers + 1 wheel, axes, limits …">

     <label for="ssc26-message">Anything else</label>
     <textarea id="ssc26-message" name="message" rows="4"></textarea>

     <button type="submit">Send</button>
   </form>

Those four questions are not idle curiosity — they are exactly the
``LawInterface`` declaration described in :doc:`the code <code>`, which is the
whole adaptation problem in one struct. Answer them and the port is mostly
already specified.

Other ways to reach us
======================

- **Email** — pmckeen@mit.edu
- **Issues and feature requests** —
  https://github.com/nscheuer/Generalized_ADCS/issues
- **Discussions** —
  https://github.com/nscheuer/Generalized_ADCS/discussions

Contributing
============

See :doc:`../contributing/index`. The library is MIT-licensed.

.. rst-class:: ssc26-privacy

   This page counts visits with `GoatCounter <https://www.goatcounter.com/>`_,
   which sets no cookies and stores no personal data. The form is handled by
   `Formspree <https://formspree.io/>`_ and reaches us by email; we use what
   you send only to answer you.
