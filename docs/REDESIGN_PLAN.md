# Documentation & README Redesign Plan

Branch: `docs/redesign` (based on `origin/main`, which is what the published site and the GitHub front page are built from; `smv_branch_whitelist = '^main$'`, `sphinx.yml` deploys only from `main`).

Status: **plan only**. Nothing in `docs/source/` or `README.md` has been changed yet.

---

## 1. Goals

1. A first-time visitor can answer three questions within one screen: *What is this? Is it for me? How do I run something?*
2. One obvious path from zero to a working simulation: **Install → Quickstart → Tutorials → Reference**.
3. A modern, consistent look on GitHub and on the docs site, in light and dark mode.
4. **No content is lost.** Every existing page, image, citation and command is kept (moved, merged or linked), and every old URL keeps working.
5. **Nothing is invented.** All claims, numbers and commands come from the existing docs, code, `pyproject.toml` or papers; every command shown is run before it is published.

## 2. What is confusing today (audit findings)

Full page-by-page inventory: 58 tutorial pages, 17 release notes, 10 satellite-factory pages, 8 installation pages, 5 contributing pages, 4 papers pages, 5 SSC26 redirect stubs, 7 legacy `docs/*.md` files, root `README.md`, `CONTRIBUTING.md`, `MAINTAINERS.md`, `ADCS/examples/README.md`.

### Getting started
- **There is no "Quickstart".** The landing page jumps from release-note cards straight into a 40-line underactuated simulation; the install page ends at `python -m ADCS.examples.basic_simulation` with no explanation of what it does or what to do next.
- **Tutorials never say how to run them.** 57 of 58 pages show a script but not its path (`examples/tutorials/<section>/<file>.py`), not the run command, and not that `examples/` only exists in a git clone (pip users don't get it). Only 6_1 and 9_3 mention running anything.
- **Two install stories disagree.** The Sphinx install page uses `.venv` + `pip install -e .`; the legacy `docs/Install_*.md` files use `venv` + `pip install -r requirements.txt`. Windows/SALTRO pages require Python 3.13, pyproject says ≥3.10, CI uses 3.12, the Linux planner build hard-codes `libpython3.12.so`.
- **The README points to the wrong place for add-on builds** ("Build instructions are in docs/Install_WSL.md and docs/Install_Windows.md"; those two files contain no build commands).
- **Links to `docs/Install_*.md` from Sphinx pages are broken on the live site** (`../../Install_X.md` relative links; the `.md` files are never deployed). `Install_macOS.md` is orphaned, yet holds the only macOS SALTRO build.

### Navigation and structure
- Landing page leads with release notes (news) instead of orientation.
- "Getting Started" sidebar caption mixes Installation, Tutorials, Release Notes and Papers; `release_notes/index.rst` also has the caption `Getting Started`.
- `tutorials/index.rst` has a "Future tutorials" section containing only the bullet `* 1_4`.
- Hub link text differs from page titles (2_0, 4_0, 6_0); section-2 filenames mix `_disturbance` and `_torque`.
- API pages (`ADCS.*.rst`) exist only after `sphinx-apidoc` runs in CI; a plain local `make html` breaks every API link. `contributing/documentation.rst` does document the apidoc step, but the Makefile doesn't run it.

### Factual errors found (fix during migration, see §7)
- `tutorials/2_7`: `:meth:` reference to `Satellite.dynamics`, which doesn't exist (`dynamics_core`, `dynamics_for_solver`, `dynamics_Hessians` do).
- `contributing/testing.rst` + `docs/Pytest.md`: example path `./test_controllers/test_controller_mtq_w_rw.py`; real path is `testing/test_controllers/mtq_w_rw/test_controller_mtq_w_rw.py`.
- `CONTRIBUTING.md` links `examples/` correctly, but README never mentions `examples/` at all.
- `docs/Install_Windows.md`: creates `venv`, then activates `.\.venv\Scripts\activate`.
- Linux trajectory-planner build type: `Release` in `.md`, `Debug` in `.rst`.
- `release_notes/0_0_1`: second code example is byte-identical to the first; "MTG" typo.
- `release_notes/0_1_2`: refers to `benchmark_planner.py --warmup`; no such file exists. Says "three new helpers", lists four.
- `release_notes/0_1_5` says the SALTRO/OldPlanner pybind clash is fixed; `Install_macOS.md` and `optional_dependencies.py` say it still occurs. **Needs maintainer confirmation; do not resolve by guessing.**
- Nested inline markup that renders as literal asterisks in 0.0.2, 0.1.2, 0.2.2.
- `papers/generalized_acs.rst` abstract is two sentences long (looks truncated); `papers/planner.rst` lists the same URL as "Paper page" and "Proceedings page".
- `tutorials/4_6` footnote cites "One Wheel Is Enough…" with no venue/year/link.
- `MAINTAINERS.md` says "Community Release: 1.0.0"; package version is 0.2.2. **Needs maintainer confirmation.**

### Weight
- `_static/release_notes/0_2_2_animation.gif` is **47 MB** and is loaded on the landing page; `0_1_0_saltro_live.gif` is 11 MB.
- 22 unused legacy images in `_static/tutorials/` (`tutorial_0N_*`), 3 unused in `_static/release_notes/`. Keep them in git history; remove from the tree only after confirming nothing external links to them.

## 3. Style direction (from trending / well-regarded repos)

Observed in the research pass (READMEs fetched via `gh api`, docs configs read from source):

| Pattern | Seen in | Use here |
|---|---|---|
| `<picture>` light/dark banner, ~880 px, linked to docs | nginx, Norman, uv (as a chart) | Yes: SVG banner with name + one-line pitch + orbit/attitude motif |
| One row of 3–5 `flat-square` shields in one accent colour | Norman, ponytail, uv, jax | PyPI version, Python versions, CI, docs build, license |
| Centred bold link row `A · B · C` | Norman, polars, jax | Docs · Install · Quickstart · Tutorials · Papers · Contributing |
| `> [!NOTE]` / `[!IMPORTANT]` alerts | nginx, Norman | Research-use disclaimer (from `contributing/about.rst`), add-ons need a clone |
| Short runnable example near the top | pydantic, jax, uv | `pip install` + `python -m ADCS.examples.basic_simulation`, then the landing-page snippet |
| Feature highlights as bold-lead bullets linking into docs | uv, polars | Replace six `<details>` blocks with a scannable grid/table |
| Citation section with BibTeX | jax, polars, Scrapling | Yes: three BibTeX entries already exist in `papers/*.rst` |
| `<details>` only for secondary material | polars, firecrawl | Optional extras table, add-on builds |
| Docs landing: sphinx-design card grid | numpy, jax, polars, scikit-learn | Yes |
| sphinx-copybutton | numpy, sklearn, jax, polars, Basilisk | Yes (already in the `docs` extra, not enabled in `conf.py`) |

Avoid (also observed): badge walls (poliastro, 16 badges), social/promo blocks above the pitch (firecrawl, pydantic), 1000+-line READMEs (minimind, DeepTutor). **Target README length: 150–250 lines.** Depth lives in the docs.

### Theme decision
**Recommendation: keep Furo** and add `sphinx-design` + `sphinx-copybutton`. Furo already handles light/dark, the existing `custom.css` (620 lines: installation picker, release cards, factory tables, PDF previews) is written against Furo's CSS variables, and `factory-tables.js` depends on it. Switching to pydata-sphinx-theme (numpy/sklearn) would mean rewriting all of that for a mostly cosmetic gain. Revisit only if a top navbar with tabs becomes a requirement.

Visual updates within Furo:
- Brand colours as Furo `light_css_variables` / `dark_css_variables` (reuse the blues already in `custom.css`: `#0066cc` / `#80c0ff`) instead of `@media` overrides.
- A project logo/wordmark for the sidebar (currently the STARLab logo is used as the project logo). Partner logos (STARLab, SSC) move to an "Affiliations" strip on the landing page and README footer.
- Replace hand-written raw-HTML cards (`installation-card`, `release-card`, tutorial cards reusing `installation-card`) with `sphinx-design` `grid`/`card`/`tab-set` so they are maintained in reST, are accessible, and survive builder changes.
- OS picker on Installation → `tab-set` with `:sync-group: os` so the choice persists across pages (replaces the custom radio + inline `<script>`).

## 4. README redesign (section by section)

Follows the `readme-design` skill layout. Source of every piece is listed; nothing new is claimed.

1. **Banner**: `assets/banner-light.svg` / `banner-dark.svg` in a `<picture>`, linked to the docs. Text: "Generalized ADCS" + pitch from the GitHub repo description. No version number in the banner.
2. **Badges** (live): PyPI version, Python versions (from classifiers: 3.10–3.12), CI (`ci.yml`), Docs (`sphinx.yml`), License MIT.
3. **Link row**: Documentation · Installation · Quickstart · Tutorials · Papers · Contributing.
4. **Pitch**: existing README paragraph ("Python framework for satellite attitude determination and control… research, prototyping, and flight-software development"), plus the audience line from `docs/source/index.rst` ("aerospace students, researchers, and engineers…").
5. **`> [!NOTE]`**: research-use disclaimer, condensed from `contributing/about.rst`.
6. **Contents** list.
7. **Hero**: existing `0_2_2_animation_preview.png` (ICESat-2), or a `<picture>` pair if a dark variant is produced.
8. **Quickstart**: `pip install generalized-adcs` → `python -m ADCS.examples.basic_simulation` (both from `installation/index.rst`), then the landing-page `ADCS.simulate(...)` snippet and its output image `_static/boston_tracking.png`. The snippet is run before publishing and the output pasted exactly.
9. **Features**: the six existing `<details>` groups become a two-column table (capability → what's included). Wording kept from the current README. Every bullet verified against the code (LQR, RK4, EKF/MEKF/UKF/SRUKF, B-dot, LP/QP allocation, factory catalog all exist). "Coming soon: Numba RK4" stays labelled as coming soon or is dropped; it is not promoted to a feature.
10. **Installation options**: table of extras taken verbatim from `pyproject.toml` (`viz`, `viz-video`, `viz-mesh`, `viz-assets`, `nrlmsise`, `ppigrf`, `dev`, `docs`, `cpp`, `poster`) with each one's comment. `[!IMPORTANT]`: `[cpp]` is source-checkout only (from the pyproject comment). Correct pointer to add-on build docs. `requirements-repro.txt` for reproducing paper campaigns.
11. **How it fits together**: one SVG diagram (light/dark): Satellite (actuators, sensors, disturbances) ↔ `simulate()` ↔ Controller / Estimator / Goal, with Orbit + Environment feeding in. Built only from module names that exist in `ADCS/` and the 2.x architecture described in `release_notes/0_2_0`. To be reviewed by a maintainer before merge.
12. **Repository map**: `ADCS/`, `examples/`, `docs/`, `testing/`, `papers/`, `tools/`, `viewer/`, SALTRO / OldPlanner submodules: what each holds.
13. **Papers & citing**: all papers from `papers/index.rst` (currently the README lists 3 of 6), with the three existing BibTeX blocks in a `<details>`. Patrick McKeen's dissertation and source-code links kept.
14. **Contributing / Community**: links to CONTRIBUTING, Code of Conduct, issues, discussions, maintainers table.
15. **Acknowledgements & license**: STARLab + SSC logos (existing), MIT.

Optional: add a `CITATION.cff` so GitHub shows "Cite this repository". Fields taken only from the existing BibTeX. **Needs maintainer OK** on which paper is the canonical citation.

## 5. New docs site structure

Principle: separate **learning** (tutorials), **doing** (how-to guides), **looking up** (reference) and **background** (papers, theory), the split used by uv, numpy and jax. URLs of existing pages are preserved wherever possible (see §6).

```
Home (index)                     ← rewritten: hero, 4-card "start here" grid, feature cards, latest release strip, affiliations
│
├─ Getting Started
│   ├─ Installation              (installation/index)       ← tab-set per OS; pip vs clone compared; extras table; verify-install step
│   ├─ Quickstart (NEW)          (getting_started/quickstart) ← basic_simulation explained line by line; landing snippet; "next steps"
│   ├─ Running the examples (NEW)(getting_started/examples)   ← where examples/ lives, clone vs curl (from ADCS/examples/README.md), run commands, Godot viewer notes from 0_0_welcome
│   └─ Core concepts (NEW)       (getting_started/concepts)   ← State / EstimatorState text moved from index.rst; Satellite, simulate(), Controller, Goal, Orbital_State: one paragraph each, linking API
│
├─ Tutorials                     (tutorials/…, unchanged paths)
│   ├─ Learning path page        ← index.rst becomes a sphinx-design card grid of the 10 sections with time/prereq badges; "Future tutorials * 1_4" removed or replaced by a real title (maintainer)
│   └─ 0_0 … 9_3                 ← every page gains a standard "Run it" box (path + command + clone requirement) and Prev/Next footers
│
├─ How-to Guides (NEW grouping of existing content)
│   ├─ Optional C++ add-ons      ← installation/trajectory_planner_* + saltro_* + content that today exists only in docs/Install_*.md (WSL setup, macOS SALTRO, lldb, co-import caveat)
│   ├─ Monte Carlo campaigns     ← link hub into tutorials 7_x + release note 0.0.1 examples
│   ├─ Hardware-in-the-loop / remote execution ← link hub into 9_3 + release notes 0.1.3/0.1.6
│   ├─ Visualization (Godot viewer) ← from release note 0.2.2 + welcome note + viz extras
│   └─ Saving & loading results  ← examples/save_and_load/* (currently undocumented scripts)
│
├─ Reference
│   ├─ API reference             (ADCS, generated)          ← grouped landing page instead of the raw apidoc tree
│   ├─ Satellite Factory         (satellite_factory/…, unchanged) ← card grid for the 9 categories; sortable tables kept
│   └─ Citations                 (contributing/citations → moved, redirect left)
│
├─ Research
│   ├─ Papers                    (papers/…, unchanged)
│   └─ Academic background       ← dissertation + source code links from README
│
└─ Project
    ├─ Release notes             (release_notes/…, unchanged) ← moved out of "Getting Started"; card grid for the latest three, list for the rest
    ├─ Contributing / Testing / Documentation (contributing/…, unchanged)
    └─ About & disclaimer        (contributing/about)
```

Components to add (`conf.py`): `sphinx_design`, `sphinx_copybutton` (with `copybutton_prompt_text` regex for `$ ` / `>>> `), optionally `sphinxext.opengraph` for link previews. Add them to `pyproject [docs]` and `docs/requirements.txt` (which are currently out of sync: `docs/requirements.txt` lacks `sphinx-copybutton` and `sphinx-autodoc-typehints`).

Build fixes: add an `apidoc` step to `docs/Makefile` (same command as CI) so local builds match CI; keep `-W --keep-going`.

## 6. Content migration map (nothing dropped)

| Existing | New home | Action |
|---|---|---|
| `README.md` (all sections) | `README.md` §4 | Restructured; papers list completed from `papers/index.rst`; wrong add-on pointer fixed |
| `docs/source/index.rst` intro, audience, "What you can do" | Home hero + feature cards | Reworded only where needed for layout |
| `index.rst` State/EstimatorState paragraph | Core concepts | Moved verbatim |
| `index.rst` example + `boston_tracking.png` | Quickstart (and Home teaser) | Moved; run and verified |
| `index.rst` release cards | Home "latest releases" strip (sphinx-design) | GIF replaced by poster image/MP4 (see §8) |
| `installation/index.rst` | same path | Raw HTML → `tab-set`; text kept |
| `installation/*_addon`, `*_linux`, `*_macos`, `*_windows` | same paths, grouped under How-to → C++ add-ons | Contradictions reconciled (with maintainer), `.md` links fixed |
| `docs/Install_WSL.md` (WSL setup, VS Code, slow-filesystem note) | new `installation/wsl.rst` | Unique content ported |
| `docs/Install_macOS.md` (SALTRO macOS build, notes, caveat, lldb) | new `installation/saltro_macos.rst` + caveat box | Unique content ported |
| `docs/Install_Windows.md` (`py -3.13 --version` check) | `installation/saltro_windows.rst` / planner Windows | Ported; venv-name bug fixed |
| `docs/Install_Trajectory_Planner.md` (`optional_dependencies` explanation, re-run note, run-examples section) | `installation/trajectory_planner_addon.rst` | Ported |
| `docs/Install_SALTRO.md` | `installation/saltro_addon.rst` | Already covered; upstream link kept |
| `docs/Install_*.md` files themselves | **kept** as short stubs pointing to the live pages | `ADCS/controller/helpers/optional_dependencies.py:67,76` prints these paths in error messages |
| `docs/Pytest.md` ("takes about 3 minutes") | `contributing/testing.rst` | Merged; wrong path fixed; file becomes a stub link |
| `docs/Sphinx.md` (`find … -delete` cleanup command) | `contributing/documentation.rst` | Merged; file becomes a stub link |
| `ADCS/examples/README.md` (clone + curl-tree download) | Getting Started → Running the examples | Linked + summarised (file stays: shipped in package) |
| `tutorials/*` (58 pages) | same paths | Add "Run it" box, Prev/Next; fix 2_7 ref; align hub link text with titles; turn inline-code pages (1_2, 1_3, 2_1, 2_2, 2_3) into `literalinclude` of their existing scripts; mention the `_godot.py` variants |
| `tutorials/index.rst` "Future tutorials: * 1_4" | Learning path page | Ask maintainer for the planned title, else remove |
| `release_notes/*` (17) | same paths, under Project | Markup fixes; duplicate 0.0.1 example flagged to maintainer; no content removed |
| `satellite_factory/*` (10) + `factory-tables.js` | same paths, under Reference | Card-grid index; tables untouched |
| `papers/*` (4) | same paths, under Research | Abstract/duplicate-link issues flagged to maintainer |
| `ssc26/*` (5 redirect stubs) | **unchanged** | Poster QR codes point here |
| `contributing/index, testing, documentation, about` | same paths | Testing/doc fixes; About also linked from footer |
| `contributing/citations.rst` | same path, also linked under Reference | No move needed; nav entry only |
| `CONTRIBUTING.md`, `MAINTAINERS.md`, `CODE_OF_CONDUCT.md`, PR/issue templates | unchanged files | README links to them; MAINTAINERS "1.0.0" flagged |
| Unreferenced example scripts (20 `examples/controllers/*`, `mc/*`, `release_notes/*`, `save_and_load/*`, legacy `tutorial_N_*.py`, 3 `*_godot.py` variants) | "Running the examples" catalogue table | Listed with one-line descriptions taken from each script's own docstring/comments only |

**URL policy:** no existing page path changes. New pages get new paths. If a later phase does move a page, leave a redirect stub using the existing `ssc26/*.rst` pattern (`:orphan:` + meta refresh), or add `sphinx-reredirects`.

## 7. Correctness pass (during migration)

Fix without asking (verifiable from the repo): 2_7 method ref; testing path; Windows venv-name mismatch; broken `../../Install_*.md` links; reST nested-markup rendering; `release_notes/index.rst` caption; 0.1.2 "three"→"four" helpers; "MTG"→"MTQ" typo; hub-link text; README add-on pointer.

**Needs a maintainer decision** (will not guess):

> Answered 2026-11-27 (applied in phase 5): (1) Release by default, Debug shown separately for GDB; (2) Windows add-on builds need Python 3.13+; (3) co-import clash reported fixed, so the caveat was not ported. Note that `ADCS/controller/helpers/optional_dependencies.py` on `main` still handles and describes the clash; (4) macOS add-on builds are community-supported.
1. Linux trajectory-planner build type: Release or Debug?
2. Required Python for Windows/SALTRO: 3.13 (pages) vs ≥3.10 (pyproject)?
3. Is the SALTRO/OldPlanner co-import clash fixed (0.1.5) or still present (macOS doc, code)?
4. Is macOS officially supported (Sphinx tab) or community-supported (`Install_macOS.md`)?
5. Intended content of 0.0.1's second example; `benchmark_planner.py` reference in 0.1.2.
6. Full abstract for `papers/generalized_acs.rst`; venue for the "One Wheel Is Enough" citation in 4_6.
7. "Community Release: 1.0.0" in MAINTAINERS.md.
8. Planned tutorial "1_4" title.
9. Canonical citation for `CITATION.cff`.

## 8. Assets

- New `assets/` at repo root: `banner-{light,dark}.svg`, `architecture-{light,dark}.svg`, generated by one script (pattern from the readme-design skill's `readme-art.example.mjs`), colours from `custom.css`, system font stacks.
- Landing-page GIF (47 MB): replace with the existing `0_2_2_animation_preview.png` poster linking to the release note, or re-encode to MP4/WebM (`<video autoplay muted loop playsinline poster=…>`), which is typically far smaller. Same treatment for the 11 MB SALTRO GIF.
- All `<img>` get descriptive alt text (several in 0.1.3 currently say "placeholder").
- Unused images (25): leave in place in this pass; list them in the PR for a later cleanup.

## 9. Phases (one PR each)

1. **README**: banner, badges, quickstart, features table, extras table, papers + BibTeX, repo map. Render-check via `gh api markdown`, screenshot light/dark.
2. **Docs foundation**: `sphinx-design`, `sphinx-copybutton`, Furo brand variables, Makefile apidoc step, requirements sync. No content changes; build must stay `-W` clean.
3. **Getting Started**: new Home, Quickstart, Running the examples, Core concepts; Installation → `tab-set`; sidebar restructure.
4. **Tutorials**: "Run it" boxes, Prev/Next, literalinclude conversion, ref fixes, learning-path index.
5. **How-to + legacy `.md` consolidation**: port unique `.md` content, stub the `.md` files, reconcile contradictions per maintainer answers.
6. **Reference/Research/Project polish**: factory card grid, API landing page, release-notes layout, papers fixes, media weight.

Each phase: `sphinx-build -W --keep-going` clean, link check (`sphinx-build -b linkcheck`), visual check in light + dark at desktop and phone widths, and every new command run on a fresh venv.

## 10. Out of scope / notes

- The `3_0/playground` branch is rebuilding the package as `GenADCS` with a new, API-only docs index. This redesign targets the published 2.x docs on `main`. The new structure (Getting Started / Tutorials / How-to / Reference) is meant to carry over to 3.0; the 3.0 docs should adopt it rather than the current API-only index.
- No content of the papers, release notes or citations is rewritten beyond markup fixes and the flagged items.

## 11. Follow-ups after phase 6

- **Remaining GIF weight:** the 29 tutorial GIFs total about 116 MB of the 153 MB `_static` folder. Re-encoding them to MP4 the same way as the two release-note animations (45 MB → 3.3 MB, 10 MB → 0.8 MB) would cut the deployed site by roughly 100 MB. They sit inside sphinx-design cards, so each needs a `<video>` block instead of `.. image::`.
- **Unreferenced files in `docs/source/_static/` (15.4 MB, 28 files).** Sphinx copies them into the deployed site anyway. Remove after checking nothing outside the docs links to them:
  `tutorials/tutorial_01_beavercube2.png`, `tutorials/tutorial_07_beavercube2.png`, `tutorials/tutorial_05_cygnss.jpeg`, `release_notes/0_1_6_raspberry_pi.jpeg`, `ground_tracking.png`, `tutorials/tutorial_06_globe.png.png`, `tutorials/tutorial_07_globe.png`, `tutorials/tutorial_05_orbitplot.png`, `tutorials/tutorial_04_plots.png`, `tutorials/tutorial_04_attitudeplot.png`, `tutorials/tutorial_03_attitudeplot.png`, `tutorials/tutorial_02_attitudeplot.png`, `tutorials/tutorial_03_plots.png`, `tutorials/tutorial_02_plots.png`, `tutorials/tutorial_01_plots.png`, `tutorials/tutorial_07_tracked_trajectory.png`, `tutorials/tutorial_05_velocityplot.png`, `tutorials/tutorial_07_planned_trajectory.png`, `tutorials/tutorial_02_swisscube.jpeg`, `tutorials/tutorial_05_positionplot.png`, `release_notes/0_1_8_smallsat_logo.png`, `papers/generalized_acs/smallsat-logo-color-social.png`, `tutorials/tutorial_03_trmm.jpeg`, `tutorials/tutorial_08_server_terminal.png`, `release_notes/0_1_7_benchmark_small.png`, `tutorials/tutorial_08_timing_terminal.png`, `tutorials/tutorial_06_beavercube1.jpg`, `papers/package/smallsatEU_logo.webp`.
- **`debug/debug_helpers/debug_animations/debug_icesat2_sunset_gif.py`** still writes `docs/source/_static/release_notes/0_2_2_animation.gif`. If the animation is regenerated, re-encode it to `0_2_2_animation.mp4` (H.264, `-movflags +faststart`) rather than committing the GIF.
- **Still open from §7:** the truncated abstract on `papers/generalized_acs.rst`, the duplicate second example in release note 0.0.1, the `benchmark_planner.py --warmup` reference in 0.1.2, the "One Wheel Is Enough" citation in tutorial 4.6, `deployment_wait_s` in tutorial 8.1, "Community Release: 1.0.0" in `MAINTAINERS.md`, and a `CITATION.cff`.
