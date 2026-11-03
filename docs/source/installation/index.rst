Installation
============

Please ensure Python>=3.10 is installed on your system.

.. raw:: html

   <div class="installation-picker" role="radiogroup" aria-label="Operating system">
     <input type="radio" name="installation-os" id="installation-linux" checked>
     <input type="radio" name="installation-os" id="installation-windows">
     <input type="radio" name="installation-os" id="installation-macos">
     <div class="installation-tabs" aria-label="Select your operating system">
       <label for="installation-linux">Linux / WSL</label>
       <label for="installation-windows">Windows</label>
       <label for="installation-macos">macOS</label>
     </div>

     <div class="installation-panels">
       <div class="installation-grid installation-linux">
         <section class="installation-card">
           <h2>GitHub Clone</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>git clone https://github.com/nscheuer/Generalized_ADCS.git
   cd Generalized_ADCS
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   .venv/bin/python -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Full examples, debug scripts, and tutorials</li>
             <li>✅ Offline documentation</li>
             <li>✅ Optional add-ons installed by default</li>
           </ul>
         </section>
         <section class="installation-card">
           <h2>pip install</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>python3 -m venv adcs-venv
   source adcs-venv/bin/activate
   pip install generalized-adcs
   python -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Lightweight setup</li>
             <li>✅ Core ADCS functionality and plotting</li>
           </ul>
         </section>
       </div>

       <div class="installation-grid installation-windows">
         <section class="installation-card">
           <h2>GitHub Clone</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>git clone https://github.com/nscheuer/Generalized_ADCS.git
   cd Generalized_ADCS
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   .\.venv\Scripts\python.exe -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Full examples, debug scripts, and tutorials</li>
             <li>✅ Offline documentation</li>
             <li>✅ Optional add-ons installed by default</li>
           </ul>
         </section>
         <section class="installation-card">
           <h2>pip install</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>python -m venv adcs-venv
   .\adcs-venv\Scripts\activate
   pip install generalized-adcs
   python -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Lightweight setup</li>
             <li>✅ Core ADCS functionality and plotting</li>
           </ul>
         </section>
       </div>

       <div class="installation-grid installation-macos">
         <section class="installation-card">
           <h2>GitHub Clone</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>git clone https://github.com/nscheuer/Generalized_ADCS.git
   cd Generalized_ADCS
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   .venv/bin/python -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Full examples, debug scripts, and tutorials</li>
             <li>✅ Offline documentation</li>
             <li>✅ Optional add-ons installed by default</li>
           </ul>
         </section>
         <section class="installation-card">
           <h2>pip install</h2>
           <div class="installation-code-box">
             <button class="installation-copy" type="button" aria-label="Copy installation commands">Copy</button>
             <pre><code>python3 -m venv adcs-venv
   source adcs-venv/bin/activate
   pip install generalized-adcs
   python -m ADCS.examples.basic_simulation</code></pre>
           </div>
           <ul class="installation-features">
             <li>✅ Lightweight setup</li>
             <li>✅ Core ADCS functionality and plotting</li>
           </ul>
         </section>
       </div>
     </div>
   </div>
   <script>
   document.querySelectorAll(".installation-copy").forEach((button) => {
     button.addEventListener("click", async () => {
       const code = button.closest(".installation-code-box").querySelector("code").innerText;
       try {
         await navigator.clipboard.writeText(code);
       } catch {
         const temporary = document.createElement("textarea");
         temporary.value = code;
         temporary.style.position = "fixed";
         temporary.style.opacity = "0";
         document.body.appendChild(temporary);
         temporary.select();
         document.execCommand("copy");
         temporary.remove();
       }
       button.textContent = "Copied!";
       window.setTimeout(() => { button.textContent = "Copy"; }, 1600);
     });
   });
   </script>

The source checkout needs Git. The PyPI path runs the same included example without downloading any additional files.

Optional C++ add-ons can be installed after the core framework. Their source
code lives in the ``OldPlanner`` and ``SALTRO`` Git submodules.

.. toctree::
   :maxdepth: 2
   :caption: Add-Ons

   trajectory_planner_addon
   saltro_addon

See :doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for an overview of
trajectory optimization add-ons and where to start.
