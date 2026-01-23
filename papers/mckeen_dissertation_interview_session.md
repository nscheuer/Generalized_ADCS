# McKeen Dissertation & Papers Interview Session
**Session Name**: mckeen_dissertation_interview
**Date**: 2026-01-23
**Purpose**: Strengthen 4 papers + dissertation through structured interview

---

## Overview

### Materials Being Reviewed
1. **Dissertation**: McKeen PhD Thesis (MIT) - Generalized ADCS
2. **Package Paper**: Generalized_ADCS Python framework (abstract submitted)
3. **3+1 Paper**: 3 MTQ + 1 RW Control (abstract submitted)
4. **Generalized Control Paper**: LTI/LTV control theory
5. **Planner Paper**: MTQ Trajectory Planner

### Codebase
- **Location**: `/home/pmckeen/Generalized_ADCS`
- **Core**: Python ADCS framework with C++ trajectory planner (ALTRO)
- **Key Contributions**: LP-based torque allocation, underactuated control, trajectory optimization

---

## Dissertation Structure (from TeX)

### Chapters Identified:
- abstract.tex
- intro.tex
- background.tex
- approach.tex
- goals.tex
- planning.tex
- simulation.tex
- estimation.tex
- disturbance.tex
- conclusion.tex
- mydesign.tex
- appendixa.tex

---

## Key Technical Understanding (from Codebase Exploration)

### Core Research Contributions:
1. **LP vs QP Torque Allocation**: LP preserves direction (17deg error vs 25.7deg for QP in closed-loop)
2. **Underactuated 3MTQ+1RW**: Time-varying controllability enables ~100% desaturation capability
3. **ALTRO Trajectory Planning**: Two-pass optimization (coarse + fine)
4. **SRUAKF Estimation**: Square-root UKF with Cholesky updates

### Key Findings:
- Direction matters more than magnitude for Lyapunov stability
- QPC constraints rarely activate (<6%)
- Reachability-aware attitude goal conversion can choose different quaternions

---

## Interview Questions & Responses

### Section 1: Big Picture & Motivation

**Q1: BeaverCube status?**
- BeaverCube had early failure, never phoned home (likely radio, not ADCS)
- BeaverCube-2 under development, using many of these ideas
- **Implication for papers**: This is simulation-validated work with flight heritage pending on BC-2. Should be framed carefully - not flight-proven yet, but designed for real mission.

**Q2: Why is 3+1 underutilized?**
- User insights:
  - Requires "weird formulations" - mathematical complexity barrier
  - Neither cheapest (that's MTQ-only) nor best performance (that's 3RW)
  - Without proper tools, hard to analyze/validate
  - Unproven → risk aversion → stays unproven (chicken-egg)

- **Research findings** (from literature search):
  - IMPORTANT FRAMING: Hybrid MTQ+RW is COMMON (3RW + 3MTQ for momentum dumping)
  - What's UNCOMMON is **reduced-wheel hybrid** (1-2 RW instead of 3)
  - Reasons for avoiding reduced-wheel:
    1. **Controllability singularities** - losing RW axes creates control allocation singularities depending on orbit, environment, attitude
    2. **Time-varying analysis required** - standard LTI tools don't apply; need LTV controllability analysis (which Patrick's work provides!)
    3. **RW failure modes** - ~50% of ADCS failures are RW-related; having only 1 RW seems risky
    4. **Testing difficulty** - hard to validate magnetic control on ground (constant field vs. orbit-varying)

  - **NARRATIVE OPPORTUNITY**: Patrick's work provides the missing tools (LTV controllability, LP allocation, trajectory planning) that make reduced-wheel architectures analyzable and trustworthy. The gap isn't "no one thought of it" - it's "no one had the tools to prove it works."

Sources:
- [Hybrid Attitude Control for Nano-Spacecraft: RW Failure and Singularity Handling](https://arc.aiaa.org/doi/10.2514/1.G005525)
- [Magnetorquer-Only Attitude Control (CMU)](https://rexlab.ri.cmu.edu/papers/magnetorquer_only.pdf)
- [Design of Attitude Control Systems for CubeSat-Class Nanosatellite](https://www.hindawi.com/journals/jcse/2013/657182/)

**Q3: Mass/power quantification?**
Research findings (from component datasheets):

| Component | Mass | Power |
|-----------|------|-------|
| Reaction Wheel (CubeSpace CW0017) | 60g | 0.4-0.7W |
| Reaction Wheel (CubeSpace CW0057) | 115g | ~0.5W |
| Magnetorquer rod (NewSpace) | 30-50g | 0.2-0.8W |
| Compact MTQ (EXA MT01) | 7.5g | 0.2W |

**Configuration comparison for 3U CubeSat:**
| Config | RW Mass | MTQ Mass | Total | vs 3+3 savings |
|--------|---------|----------|-------|----------------|
| 3+0 | 0g | ~90g | ~90g | 300g (77%) |
| 3+1 | ~100g | ~90g | ~190g | 200g (51%) |
| 3+3 | ~300g | ~90g | ~390g | baseline |

**Key claim for papers**:
- 3+1 saves ~200g vs 3+3 = **5% of entire 4kg 3U mass budget**
- Power savings: ~0.8-1.4W continuous
- This is significant for mass/power-constrained missions

Sources:
- [CubeSpace Reaction Wheels](https://www.satcatalog.com/component/ku-leuven-rw/)
- [EXA MT01 Magnetorquer](https://satsearch.co/products/exa-mt01-compact-magnetorquer)
- [NanoAvionics MTQ](https://nanoavionics.com/cubesat-components/cubesat-magnetorquer-satbus-mtq/)

**Q4: Target audience & publication strategy?**
Building "mega-papers" to be cut down for 2 venues each:

| Paper | Conference | Journal Target | Audience Implication |
|-------|------------|----------------|---------------------|
| 3+1 Paper | SmallSat Europe | JOSS | Practitioners → OSS community |
| Package Paper | SmallSat Europe | JOSS | Practitioners → OSS community |
| Generalized Control | SmallSat USA | JGCD | Practitioners → Academic GNC |
| Planner Paper | SmallSat USA | JGCD | Practitioners → Academic GNC |

**Framing implications:**
- SmallSat versions: Emphasize practical results, mission applicability, "here's what you can do"
- JOSS versions: Emphasize code quality, documentation, reproducibility, ease of use
- JGCD versions: Emphasize theoretical contributions, proofs, novel formulations

**Note**: JGCD reviewers will want rigorous proofs and comparison to prior art. JOSS reviewers will want clean API, tests, and documentation.

### Section 2: Technical Contributions

**Q5: LP vs QP - why does LP outperform?**
- STATUS: Open research question
- User insight: "Constrained QP should perform at least as well as LP because QP contains the LP solution"
- Empirical finding: QP does worse (25.7° vs 17° in closed-loop)
- WORKING ON: Finding constraints that prevent QP from underperforming
- **GAP FOR PAPER**: This needs resolution before Generalized Control paper is complete. Either:
  1. Explain theoretically why unconstrained/poorly-constrained QP fails, OR
  2. Find QP constraints that match LP performance, OR
  3. Prove LP is fundamentally better for this application

**Q6: 100% vs 96% discrepancy**
- 3+1 paper: newer data, shorter timescale → "100% within 1 degree"
- Planner paper: older data → "96% sub-degree"
- ACTION: Update Planner paper data to match 3+1 methodology for consistency

**Q7: Dissertation contributions → Paper mapping**

**The 5 Dissertation Contributions:**
1. Generalized Disturbance- & Dynamics-Aware Estimator
2. Direct Countering of Disturbances
3. General Trajectory Planner
4. Improved ADCS Autonomy
5. ADCS Architecture for Weak & Underactuated Satellites

**Proposed Mapping:**

| Contribution | Primary Paper | Secondary Paper(s) |
|--------------|---------------|-------------------|
| 1. Estimator | Package Paper (framework) | — (no dedicated paper?) |
| 2. Disturbance Countering | Generalized Control | 3+1 (Monte Carlo w/ disturbances) |
| 3. Trajectory Planner | **Planner Paper** | 3+1 (planner-driven results) |
| 4. ADCS Autonomy | Package Paper | All (philosophy throughout) |
| 5. Underactuated Architecture | **3+1 Paper** | Generalized Control (LP, controllability) |

**Observations:**
- **Estimator (Contribution 1)** has no dedicated paper. Is this intentional? The SRUAKF is a significant contribution (0.01° accuracy claim). Could be its own paper or featured more prominently in Package Paper.
- **Package Paper** carries Contributions 1 & 4 but also serves as "glue" for everything
- **Generalized Control Paper** is the theoretical foundation (LP allocation + controllability analysis) that enables Contributions 2 & 5
- **3+1 Paper** is the flagship demonstration of Contribution 5 in action
- **Planner Paper** is a clean 1:1 mapping to Contribution 3

**Gap identified**: Disturbance estimation/countering is spread across papers but not the star of any single one. Is this okay, or should it be consolidated?

**Q8: Graceful degradation - concrete behavior**
User description:
- No wild attempts to break limits
- No spinning out of control
- Gets as close as possible (when achievable)
- Drifts gracefully respecting limits
- Sets up for next goal
- Behavior depends on control law and operator goals
- Core philosophy: "Get as close as possible while respecting constraints"

**NEED**: A specific worked example for papers (e.g., "Operator requests nadir pointing but sun angle constraint makes it infeasible. System does X instead of Y.")

**Q9: Reduced-attitude vs full-attitude**
User explanation:
- Reduced attitude = point body vector at world vector, no constraint on rotation about that axis
- Add damping to prevent excessive spin about pointing axis
- Works for: cameras, solar panels, radios, antennas
- What you give up:
  - Less complete control of orientation
  - Less predictable final state during mission design
  - Potential angular velocity about pointing axis (problem for high-precision imaging)
- User's view: "Doesn't seem like much unless JWST/Hubble level or 2 simultaneous pointing goals"
- Research findings on potential downsides:
  1. **De-spin complexity**: If antennas/instruments need stable pointing while body rotates, may need de-spin mechanisms
  2. **Thermal asymmetry**: Rotation about pointing axis may cause uneven heating/cooling
  3. **Multiple payload conflicts**: If you have 2+ instruments on different body axes, can't satisfy both simultaneously
  4. **Knowledge vs accuracy gap**: You know boresight direction precisely but less certainty about full 3D pose
  5. **Constraint handling**: Adding keep-out zones (e.g., "don't point sensor at sun") is geometrically harder

  Sources: [Adaptive Reduced Attitude Control](https://ieeexplore.ieee.org/document/10004997/), [Two-Axis Magnetic Attitude Control](https://cdcl.umd.edu/papers/scitech24b.pdf)

**Q10: Two-pass trajectory optimization**
- Pragmatic reason: converged faster during development
- Allows including disturbances in second pass only (faster/more stable?)
- **GAP**: Needs more rigorous justification for papers. Why does two-pass converge better? Is it always better or just for certain scenarios?

**Q11: 100 Monte Carlo runs**
- User view: "Seems like a lot, plots show consistency"
- Could run more if needed
- **NOTE FOR JGCD**: May need statistical justification (confidence intervals, convergence analysis). 100 runs gives ~10% margin of error at 95% confidence for binary outcomes.

**Q12: Spin for disturbance rejection**
Two interpretations:
1. Gyroscopic stability (higher angular momentum = more stable)
2. **KEY FINDING**: Planner autonomously discovered spinning around propulsion axis to smear body-fixed off-axis thrust disturbance, preventing accumulation in any one direction
   - User had already thought of this, but planner found it independently
   - **NARRATIVE GOLD**: This demonstrates the power of autonomous trajectory optimization—it discovers clever solutions without being explicitly told

**ACTION**: Verify this spin-smearing result is in the Planner paper. If not, it MUST be added. This is a compelling demonstration of autonomy.

### Section 3: Evidence & Validation

**Q13: Package Paper TODOs - Priority Assessment**

FOR SMALLSAT EUROPE (must-haves):
1. **SMALLSAT-1**: "5-minute demo" case study - CRITICAL. Shows accessibility.
2. **SMALLSAT-2**: Practitioner metrics (sim speed, memory) - HIGH. "1 orbit in 30 seconds" type claims.
3. **TODO-DATA-2**: Comparative case study (lines of code savings) - HIGH. "5 lines vs 500 lines" is compelling.

FOR SMALLSAT EUROPE (nice-to-haves):
- SMALLSAT-3: Debugging guidance (can be brief/appendix)
- TODO-DATA-1: Framework benchmarks (timing) - good but not essential for conference

FOR JOSS (defer to journal submission):
- JGCD-1: Formal Basilisk comparison - essential for journal, not conference
- JGCD-2: Estimation error bounds derivation
- JGCD-3: Stability/convergence proofs
- TODO-DATA-7: HIL validation (nice demo, but Raspberry Pi photo may suffice for conference)

**Feasibility assessment**:
- SMALLSAT-1, SMALLSAT-2: ~1-2 days work, just run simulations and document
- TODO-DATA-2: ~1 day, straightforward comparison
- Basilisk comparison: ~1 week minimum, requires learning Basilisk API

**RECOMMENDATION**: Focus on SMALLSAT-1 + SMALLSAT-2 + TODO-DATA-2 for conference. Save Basilisk comparison for journal.

---

**Q14: QPC <6% activation - interpretation**
- STATUS: Open question, unclear if good or bad
- Possible interpretations:
  - GOOD: System rarely needs constraints → well-designed nominal operation
  - BAD: Constraints too loose → not actually protecting anything
  - NEUTRAL: QPC is a fallback for edge cases, working as intended
- **NEEDS**: Clarification in paper on what QPC is supposed to do and when

---

**Q15: 0.01° estimator accuracy validation**
- Simulation-validated with noise, bias, disturbances
- No flight truth data yet (BeaverCube-2 pending)
- **For papers**: Be precise about what "accuracy" means:
  - 0.01° RMS error? Peak? 3-sigma?
  - Against what truth? Propagated truth state?
  - Over what duration/conditions?

---

**Q16: 3+1 vs 3+3 Break-Even Analysis Scenarios**

Proposed test matrix:

| Scenario | MTQ Strength | RW Size | Orbit | Goal Type | Hypothesis |
|----------|--------------|---------|-------|-----------|------------|
| 1. Baseline | Standard 3U (0.2 Am²) | Small (2 mNms) | 500km LEO | Nadir pointing | 3+1 sufficient |
| 2. Weak MTQ | Half-strength (0.1 Am²) | Small | 500km LEO | Nadir | 3+1 may struggle |
| 3. Large slew | Standard | Small | 500km | 90° slew in 60s | Time-constrained → needs 3+3? |
| 4. High inclination | Standard | Small | 98° SSO | Nadir | Magnetic field changes → test LTV |
| 5. High precision | Standard | Small | 500km | 0.1° pointing | 3+3 likely needed |
| 6. High precision + planner | Standard | Small | 500km | 0.1° pointing | Can 3+1 + planner match 3+3? |
| 7. GEO-like weak field | Standard | Small | 800km | Nadir | Weaker B → test MTQ limits |
| 8. Large RW | Standard | Large (10 mNms) | 500km | Nadir | Does bigger RW help 3+1? |
| 9. Disturbance-heavy | Standard | Small | 400km (high drag) | Nadir | Disturbances dominate |
| 10. Fast retargeting | Standard | Small | 500km | 5 targets/orbit | Agility test |

**Key questions to answer**:
- What pointing accuracy threshold separates 3+1 from 3+3?
- Does the planner close the gap at high precision?
- Which parameter (MTQ strength, RW size, orbit, goal type) matters most?

---

**Q17: Trajectory Planning Comparison - Prior Art**

Traditional slew planning approaches to compare against ALTRO:

| Method | Description | Pros | Cons | Reference |
|--------|-------------|------|------|-----------|
| **Eigenaxis + trapezoidal** | Rotate about shortest-path axis with trapezoid velocity profile | Simple, analytical, fast | No constraints, no disturbances, rest-to-rest only | [STK Modes](https://help.agi.com/stk/) |
| **Polynomial shaping** | Fit quaternion trajectory with polynomials, use inverse dynamics | Smooth torque profiles, arbitrary BCs | No online replanning, constraint checking is post-hoc | [ResearchGate](https://www.researchgate.net/publication/280737741) |
| **RRT\*/A\*** | Sample-based or graph search through attitude space | Handles complex constraints | Slow, discrete, needs smoothing | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0273117723003022) |
| **Potential functions** | Repulsive barriers for keep-out, attractive for goal | Works with PD control | Local minima, tuning-sensitive | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9784449/) |
| **Convex SDP** | Cast as semidefinite program for optimal fixed-time trajectory | Globally optimal, fast | Limited constraint types | [USU SmallSat](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4436&context=smallsat) |

**Suggested comparison for Planner paper**:
1. **Eigenaxis + trapezoidal** as baseline (what most missions use today)
2. **Polynomial shaping** as "smarter but still offline" comparison
3. **ALTRO** as proposed method

**Metrics to compare**:
- Computation time (offline vs realtime)
- Constraint satisfaction (keep-out violations)
- Fuel/momentum cost
- Settling time
- Robustness to disturbances
- Ability to handle infeasible goals gracefully

**Key ALTRO advantages to highlight**:
- Online replanning capability
- Native constraint handling (not post-hoc checking)
- Discovers non-obvious solutions (spinning for disturbance rejection)
- Handles underactuated systems (not assumed in most methods)
- Disturbance-aware planning

---

**Keep-out zone avoidance - prior art** (user asked how others do it)

Main approaches in literature:
1. **Hand-planned slews** - Ground operators design waypoints avoiding constraints (common for high-value missions)
2. **Geometric waypoints** - Compute intermediate attitudes outside exclusion cones, then slew point-to-point
3. **Potential functions** - Add repulsive terms to Lyapunov controller (logarithmic barriers)
4. **Reference governor** - Modify reference trajectory online to stay in safe set
5. **Deep RL** - Emerging approach, not yet proven reliable

User note: Reduced-attitude slew with keep-out zone already published (JGCD) and in dissertation. This is prior art to cite!

Sources:
- [Feedback control for eigenaxis rotations](https://arc.aiaa.org/doi/abs/10.2514/3.21555)
- [Sun-Avoidance Slew Planning](https://arc.aiaa.org/doi/10.2514/1.A34671)
- [Potential function methods](https://www.sciencedirect.com/science/article/abs/pii/S127096382300634X)

### Section 4: Narrative & Impact

**Q18: One-sentence takeaway for 3+1 paper**
User draft: "We can do <complex mission with good pointing> with only one reaction wheel—cheaper, easier, smaller."

**NEEDS**: A concrete mission example. Options:
- "Earth observation with sub-degree pointing using one reaction wheel instead of three"
- "Optical inter-satellite links with 3+1 architecture"
- "LEO imaging constellation at half the ADCS mass"

Suggested polished version:
> "Sub-degree Earth observation pointing—traditionally requiring three reaction wheels—is achievable with a single reaction wheel and three magnetorquers, cutting ADCS mass in half."

**ACTION**: Pick a specific mission class for the takeaway. Earth observation CubeSat seems most relatable to SmallSat audience.

---

**Q19: Autonomy level**
User: "Supervised autonomous to fully autonomous"

**Clarify for papers—what does operator still provide?**
- Goals (pointing targets, times)
- Constraints (keep-out zones, rate limits)
- Satellite parameters (inertia, actuator specs)

**What does the system do autonomously?**
- Plans trajectories
- Handles infeasible goals gracefully
- Adapts to disturbances
- Manages momentum
- Discovers non-obvious solutions (spinning)

**What it does NOT do (yet)?**
- Self-diagnose hardware failures?
- Adjust goals based on mission priority?
- Learn from experience?

**Suggested framing**: "Autonomous execution with operator-defined objectives"—the satellite handles *how* to achieve goals, operator defines *what* goals to achieve.

---

**Q20: Failure modes by paper**

| Paper | Component | Failure Mode | Consequence | Mitigation |
|-------|-----------|--------------|-------------|------------|
| **3+1** | MTQ sizing | MTQs too weak for orbit/disturbance | Can't desaturate RW, momentum accumulates | Size MTQs for worst-case B-field |
| **3+1** | Orbit selection | High altitude = weak B-field | MTQ authority drops, 3+1 may become 0+1 | Validate B-field strength for orbit |
| **3+1** | RW failure | Single RW fails | Degrades to MTQ-only (much worse) | No redundancy—this is the tradeoff |
| **Package** | Misconfiguration | Wrong inertia matrix, sensor noise params | Estimator diverges, controller unstable | Validation tools, sanity checks |
| **Package** | Python speed | Real-time loop too slow | Missed control deadlines | Profile, optimize hot paths, or use C++ planner |
| **Gen. Control** | LP infeasible | Requested torque impossible | Allocation returns closest feasible | By design—graceful degradation |
| **Gen. Control** | LTV assumption | Orbit doesn't vary B-field enough | Controllability claims invalid | Check eigenvalues over orbit |
| **Planner** | ALTRO divergence | Bad initial guess, aggressive goal | Planner returns no solution | Warm-start from previous, relax goal |
| **Planner** | Model mismatch | Actual inertia ≠ planned | Tracking error accumulates | TVLQR/MPC corrects online |

**RECOMMENDATION**: Add a "Limitations" or "When Not to Use" section to each paper. Reviewers appreciate honesty about boundaries.

---

**Q21: BeaverCube-2 timeline**
- JGCD submission: ~1 year out
- Flight results before submission: Unlikely
- User no longer directly on BC-2

**Framing for papers**:
- "Flight software developed and tested in HIL environment"
- "Deployed on BeaverCube-2 mission (launch pending)"
- Do NOT claim flight-proven until it actually works on orbit

---

**Q22: Generalization limits**

| Configuration | Works? | Notes |
|---------------|--------|-------|
| Standard 3RW + 3MTQ | ✅ Yes | Baseline case |
| 3+1 (3MTQ + 1RW) | ✅ Yes | Main contribution |
| MTQ-only | ✅ Yes | Demonstrated |
| CMGs | ⚠️ Partial | Implementation ongoing, not complete |
| Large flexible structures | ⚠️ Partial | ALTRO extended, but not in current package |
| Deep space (no B-field) | ❌ No | MTQ-based methods don't apply |
| Large satellites | ❌ No | MTQs too weak for large inertias |
| Thrusters only | ❓ Unknown | Should work with allocation layer? |

**For papers**: Be explicit about scope. "This work focuses on small satellites in LEO with magnetic and reaction wheel actuation. Extension to CMGs and flexible structures is future work."

---

## Notes Per Paper

### Package Paper
**Status**: Abstract submitted
**Focus**:
**Gaps identified**:
**Evidence needed**:

### 3+1 Paper
**Status**: Abstract submitted
**Focus**:
**Gaps identified**:
**Evidence needed**:

### Generalized Control Paper
**Status**:
**Focus**:
**Gaps identified**:
**Evidence needed**:

### Planner Paper
**Status**:
**Focus**:
**Gaps identified**:
**Evidence needed**:

---

## Action Items
(To be populated after interview)

---

## CRITICAL: Prior Work to Cite (3+1 Architecture)

**You are NOT the first to propose 3MTQ + 1RW!** Found two papers:

### 1. AIAA SciTech 2023
**"Attitude Control of a 3U CubeSat with Combination of Magnetorquers and Reaction Wheels"**
- DOI: [10.2514/6.2023-0935](https://arc.aiaa.org/doi/10.2514/6.2023-0935)
- Proposes PD controller with 3MTQ + 1RW (pitch axis)
- Compares to MTQ-only and 3RW
- **You MUST cite this and differentiate**

### 2. Li et al. 2013 (Journal of Control Science and Engineering)
**"Design of Attitude Control Systems for CubeSat-Class Nanosatellite"**
- [Hindawi](https://www.hindawi.com/journals/jcse/2013/657182/)
- Uses 3MTQ + 1RW for "stable mode"
- Claims 0.02° accuracy with PD magnetic control
- **This is older but relevant prior art**

### 3. JGCD 2020 - Hybrid Attitude Control
**"Hybrid Attitude Control for Nano-Spacecraft: Reaction Wheel Failure and Singularity Handling"**
- [DOI: 10.2514/1.G005525](https://arc.aiaa.org/doi/10.2514/1.G005525)
- Addresses 3RW + 3MTQ with RW failure scenarios
- Not exactly 3+1 by design, but related failure-mode work

---

## How to Differentiate Your 3+1 Paper

Your contributions BEYOND prior work:
1. **Comprehensive Monte Carlo validation** (100 runs, varied conditions) vs single-case studies
2. **Trajectory planning with ALTRO** - prior work uses PD only
3. **LP allocation** with direction-preserving properties
4. **Graceful degradation** philosophy for infeasible goals
5. **LTV controllability analysis** proving why 3+1 works
6. **Planner-discovered solutions** (spinning for disturbance rejection)
7. **Quantitative comparison** across 3+0, 3+1, 3+3 with same disturbances

**Suggested framing**: "While 3+1 architectures have been proposed [cite AIAA 2023, Li 2013], prior work lacks comprehensive analysis of controllability, trajectory planning, and graceful degradation. We provide the first systematic evaluation of 3+1 performance bounds and demonstrate that with proper planning, 3+1 can match 3+3 performance for many mission profiles."

---

## Detailed Prior Work Analysis

### AIAA 2023 Paper Details:
- **Mission**: 3U CubeSat deploying picosat via electrodynamic tether
- **Approach**: PD controller with 3MTQ + 1RW on pitch axis
- **Testing**: MATLAB simulation to determine gains and disturbance tolerance
- **Limitations**: Appears to be single-case or limited testing, PD only (no planner), no Monte Carlo
- **Full results behind paywall** - Patrick should obtain and read

### AIAA 2023 Paper - FULL ANALYSIS (Zhu et al., U Michigan)

**Paper**: "Attitude Control of a 3U CubeSat with Combination of Magnetorquers and Reaction Wheels"
**Mission**: MiTEE-II - 3U CubeSat deploying picosat via 30m electrodynamic tether, needs nadir pointing

**What they did**:
- Tested 3 schemes: (1) LQR + 3MTQ, (2) PD + 3RW, (3) PD + 3MTQ + 1 pitch-axis RW
- MATLAB simulation only
- Single case per configuration (NOT Monte Carlo)
- Conservative initial conditions for 3+1: [φ θ ψ] = [-2, -2, 2]° and [ω] = [-0.1, 0.1, -0.1]°/s
- Small disturbances: ±2×10⁻⁷ Nm for 3+1 (100× smaller than Patrick's tests)

**Their 3+1 results**:
- "Up to 0.5-degree-magnitude offset in pitch direction"
- Can handle 20× more disturbance than MTQ-only
- Acknowledges RW will saturate; desaturation "future work"

**Their limitations (explicitly stated in paper)**:
- "Monte-Carlo simulation is recommended for control authority evaluation" (they didn't do it!)
- "Desaturation procedure should be implemented to the controller in the future"
- No trajectory planning
- No reduced-attitude goals
- Mission-specific (tether deployment), not general

---

### COMPARISON TABLE: AIAA 2023 vs McKeen

| Aspect | AIAA 2023 (Zhu et al.) | McKeen (Your Work) |
|--------|------------------------|-------------------|
| **Testing** | Single case, conservative ICs | 100-run Monte Carlo, varied ICs/orbits/goals |
| **Initial conditions** | Tiny: 2° attitude, 0.1°/s rate | Large: includes 180° slews |
| **Disturbances** | Random ±2×10⁻⁷ Nm | Comprehensive modeling + feedforward compensation |
| **Control law** | PD only | PD + LP allocation + ALTRO planner |
| **Goal types** | Fixed nadir pointing only | Full-attitude, reduced-attitude, set-based, time-varying |
| **Infeasible goals** | Not addressed | Graceful degradation philosophy |
| **Controllability** | References Yang 2016 | Original LTV analysis with torque polytopes |
| **Desaturation** | "Future work" | Implemented and analyzed |
| **RW configuration** | Pitch-axis only (mission-specific) | Flexible, any axis |
| **Accuracy claimed** | 0.5° offset (steady-state, small ICs) | 2.3° mean (slews w/ disturbances), 0.05° with planner |
| **Simulation duration** | Short, single scenario | 1000s, multiple targets |
| **Allocation** | Simple torque split | LP with direction preservation |
| **Code/reproducibility** | Not released | Open source Python framework |

**YOUR KEY DIFFERENTIATORS (use these in paper)**:
1. ✅ Monte Carlo validation (they recommend it; you did it)
2. ✅ Trajectory planning (they have none)
3. ✅ Graceful degradation (they have none)
4. ✅ LTV controllability analysis (they just cite Yang)
5. ✅ Comprehensive disturbance handling (feedforward, modeling)
6. ✅ General framework vs mission-specific solution
7. ✅ Open source, reproducible

**Suggested citation framing**:
> "Zhu et al. [AIAA 2023] demonstrated a 3MTQ+1RW configuration for the MiTEE-II mission using PD control, achieving 0.5° steady-state pointing under conservative initial conditions. Our work extends this by providing (1) rigorous Monte Carlo validation across diverse scenarios, (2) trajectory planning that enables sub-degree pointing during slews, (3) graceful degradation for infeasible goals, and (4) LTV controllability analysis proving when 3+1 architectures are theoretically viable."

---

### Li 2013 Paper Details:
- **Satellite**: 1U CubeSat (smaller than Patrick's 3U)
- **Approach**: Two-stage coarse/fine control, sliding-mode + fuzzy logic
- **Testing**: Numerical simulation + AIR-BEARING HARDWARE TESTS (impressive!)
- **Accuracy**: 0.8° steady-state with sliding-mode (vs 5° for PID)
- **Focus**: Fault-tolerant control (RW friction, saturation, noise, dead zones, bias)
- **Not 0.02°**: Earlier search result may have been wrong; 0.8° is the stated accuracy

### Q24 Analysis: 0.02° vs 2.3° Discrepancy
The Li 2013 claim was actually **0.8° steady-state** (I misread earlier search result).
- Li 2013: 0.8° steady-state, 1U, sliding-mode control, air-bearing hardware
- Patrick 3+1: 2.3° mean over slews with disturbances, 3U, PD control, 1000s Monte Carlo
- Patrick 3+1 + planner: 0.05° mean

**These are NOT comparable scenarios**:
- Steady-state holding vs active slewing/pointing at varying targets
- 1U vs 3U (different inertias)
- Different disturbance environments
- Different control laws (sliding-mode vs PD vs planner)
- Different test durations and conditions

**Recommendation**: When citing Li 2013, note the different test conditions. Your Monte Carlo with varied disturbances and slew scenarios is more comprehensive.

---

## Your Key Differentiators (for all papers)

| Aspect | Prior Work | Your Work |
|--------|------------|-----------|
| Testing | Single cases, limited conditions | 100-run Monte Carlo, varied initial conditions, orbits, goals |
| Control | PD, LQR, sliding-mode | PD + LP allocation + ALTRO planner |
| Goals | Fixed pointing | Full-attitude, reduced-attitude, set-based goals |
| Infeasibility | Not addressed | Graceful degradation philosophy |
| Controllability | Assumed or not analyzed | LTV controllability analysis, torque polytopes |
| Disturbances | Limited or none | Comprehensive disturbance modeling + feedforward |
| Solutions | Hand-designed | Planner discovers non-obvious solutions (spinning) |
| Validation | Single sim or single hardware | Extensive sim + HIL + planned flight (BC-2) |

---

## Additional TODOs Identified

### For 3+1 Paper:
- [ ] **CITE PRIOR WORK** - AIAA 2023, Li 2013, JGCD 2020
- [ ] Add explicit differentiation section ("Our contributions beyond prior work")
- [ ] Break-even analysis (when does 3+1 fail vs 3+3?)
- [ ] Mass/power table with real component specs

### For Package Paper:
- [ ] 5-minute demo case study (SmallSat priority)
- [ ] Lines-of-code comparison table
- [ ] "When NOT to use this framework" section
- [ ] Installation quickstart

### For Generalized Control Paper:
- [ ] Resolve LP vs QP mystery (why does LP win?)
- [ ] Add stability proof for allocation layer
- [ ] Clarify QPC <6% finding

### For Planner Paper:
- [ ] Comparison to eigenaxis + trapezoidal baseline
- [ ] Computational cost table (ALTRO vs alternatives)
- [ ] Sensitivity analysis (marked [TBD] in paper)

### Cross-cutting:
- [ ] Consistent data between papers (100% vs 96% discrepancy)
- [ ] Explicit "Limitations" section in each paper
- [ ] BeaverCube-2 framing: "flight-ready, launch pending" not "flight-proven"

---

## Potential Reviewer Critiques & Preemptive Responses

### 3+1 Paper

| Critique | Severity | Preemptive Response |
|----------|----------|---------------------|
| "Why use this weird architecture? Just use 3 RWs." | HIGH | Quantify mass/power savings (200g, 5% of 3U budget). Not all missions can afford 3RW. Some missions are mass-constrained. Constellation economics favor cheaper units. |
| "Just relax your pointing requirements instead" | MEDIUM | Some missions have hard requirements (optical crosslinks, specific science). 3+1 lets you MEET requirements, not relax them. |
| "No flight heritage, unproven" | HIGH | BeaverCube-2 flight pending. HIL validated. Prior work exists (cite AIAA 2023, Li 2013). Simulation is high-fidelity with realistic disturbances. |
| "100 Monte Carlo runs isn't enough" | MEDIUM | Show convergence plots (results stabilize well before 100). Offer to run more. Note that prior work uses single cases. |
| "Prior work already did this (AIAA 2023)" | HIGH | Differentiate: Monte Carlo validation, ALTRO planner, LP allocation, graceful degradation, LTV analysis. Prior work is PD-only, limited testing. |
| "RW failure makes this worse than MTQ-only?" | MEDIUM | Acknowledge single-point-of-failure tradeoff. For missions prioritizing performance over redundancy. Future work: fault-tolerant mode switching. |

### Package Paper

| Critique | Severity | Preemptive Response |
|----------|----------|---------------------|
| "There are other options (Basilisk, etc.)" | HIGH | Basilisk is C++ with Python wrappers, message-passing architecture, harder to modify. We searched for comparable pure-Python ADCS packages and found none with closed-loop simulation. Table comparing features. |
| "Python is too slow for flight software" | MEDIUM | HIL tests on Raspberry Pi demonstrate sufficient performance. Hot paths can be optimized or moved to C++. Many CubeSats already use Python. Development speed > runtime speed for many missions. |
| "ADCS is a small part of satellite—why focus here?" | LOW | ADCS failures are a leading cause of mission failure. Better tools reduce development time and bugs. Framework enables rapid architecture trades. |
| "No formal verification/proofs" | MEDIUM | Extensive unit tests, integration tests, Monte Carlo validation. Formal proofs deferred to Generalized Control paper. |
| "Why not contribute to Basilisk instead?" | MEDIUM | Different design philosophy (pure Python, modular, configurable). Basilisk's architecture makes major changes difficult. This is a complement, not replacement. |

### Planner Paper

| Critique | Severity | Preemptive Response |
|----------|----------|---------------------|
| "Planners are scary and can fail" | HIGH | Graceful degradation: infeasible goals produce bounded, predictable behavior. Fallback to PD if planner fails. Real-world implementation details included. TVLQR/MPC tracking handles model mismatch. |
| "Other planners exist (MPC, etc.)—why ALTRO?" | HIGH | Compare to eigenaxis/polynomial baselines. ALTRO handles constraints natively, discovers non-obvious solutions. Comparison table needed. |
| "Computational cost too high for flight" | MEDIUM | Two-pass optimization reduces cost. Coarse pass fast, fine pass only when needed. Offline planning with online tracking is viable. Include timing benchmarks. |
| "Disturbance model must be accurate" | MEDIUM | TVLQR/MPC corrects for model mismatch online. Sensitivity analysis shows robustness to model errors. Second pass includes disturbances. |
| "Spinning solution is a special case, not general" | LOW | It's an EXAMPLE of planner discovering solutions. The point is the planner finds things humans might miss. Other examples likely exist. |

### Generalized Control Paper

| Critique | Severity | Preemptive Response |
|----------|----------|---------------------|
| "This is a lot of math for not much practical gain" | HIGH | LP allocation: 17° vs 25.7° error (significant!). LTV controllability enables 3+1 analysis. Math enables tools that enable missions. |
| "LP vs QP finding is unexplained" | HIGH | **OPEN ISSUE** - needs resolution before submission. Either explain why or find QP constraints that work. |
| "Stability proofs are missing" | HIGH | Add Lyapunov analysis for allocation layer. Prove LP direction preservation maintains stability. |
| "QPC <6% means constraints don't matter" | MEDIUM | Clarify what QPC does. Low activation may mean system is well-designed for nominal, but constraints provide safety net for edge cases. |
| "LTV analysis assumes known B-field" | MEDIUM | Magnetic field models (IGRF) are well-characterized. Sensitivity analysis to field errors. Estimator can adapt to measured field. |

---

## Biggest Single Weakness Per Paper (Patrick's Q25)

1. **3+1 Paper**: Prior work exists (AIAA 2023) and you need to clearly differentiate. Also, no flight heritage yet—simulation-only.

2. **Package Paper**: Lack of direct comparison to alternatives. "We couldn't find comparable packages" needs documentation—show the search, explain why Basilisk/OpenSatKit don't qualify.

3. **Planner Paper**: Computational cost and failure modes. Reviewers will ask "what happens when ALTRO fails?" Need explicit fallback strategy and timing benchmarks.

4. **Generalized Control Paper**: The LP vs QP mystery is unresolved. This is the central theoretical claim and it's not fully explained yet. JGCD will not accept hand-waving here.

---

## User's Self-Identified Weaknesses (Q25 response)

- **3+1**: "Weird architecture, why go through control games—just relax goals or get RWs"
- **Package**: "Other options exist" (but couldn't find comparable ones), "ADCS is small part of mission"
- **Planner**: "Planners are scary and can fail", "Other planners exist, why this one?"
- **Generalized Control**: "Lot of math for not much reason"

These align with reviewer critiques above. Address all of them explicitly in the papers.

---

## Refined Narrative (Q27)

### From Thesis Abstract - Key Themes:
1. **Generalizability**: "adapts automatically to different satellite types, mission requirements, and operational goals"
2. **Autonomy**: "hands-off," "calculating its own slews and desaturation," "reducing reliance on predefined ground-based commands"
3. **Enabling weak systems**: "weaker or fewer actuators," "commercial off-the-shelf components in high-performance missions"
4. **Graceful performance**: "even in cases of underactuation or large disturbances"

### Proposed Unified Narrative:

> **"Modern small satellites shouldn't have to choose between capability and cost. This work provides the tools—estimation, control, allocation, and planning—that let autonomous ADCS gracefully achieve high performance with limited hardware. The satellite adapts to what it has, not what we wish it had."**

### One-liner versions for each paper:

| Paper | Narrative Hook |
|-------|----------------|
| **3+1** | "Sub-degree pointing with one reaction wheel—proving the 'sweet spot' architecture actually works." |
| **Package** | "Configure, don't code: a Python framework that makes advanced ADCS accessible to any small sat team." |
| **Planner** | "When the goal is impossible, don't fail—adapt. Trajectory planning that discovers what your satellite can actually do." |
| **Gen. Control** | "Direction matters more than magnitude: the math behind graceful torque allocation." |

### How papers connect:

```
Dissertation Vision: Autonomous, Graceful ADCS for Limited Systems
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   [Gen. Control]        [Planner]            [Package]
   "How to allocate"   "How to plan"      "How to build it"
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                          [3+1 Paper]
                    "Proof it works in practice"
```

The **Generalized Control** paper provides the theoretical foundation (LP allocation, LTV controllability).
The **Planner** paper shows how to exploit that foundation for trajectory optimization.
The **Package** paper makes it all accessible and usable.
The **3+1** paper is the flagship demonstration that the whole system works.

---

## Specific Arguments & Evidence Against Each Weakness

### 3+1 Paper Weaknesses

| Weakness | Existing Arguments | Evidence to Gather |
|----------|-------------------|-------------------|
| "Weird architecture—just use 3 RWs" | Mass/power savings (200g = 5% of 3U budget). Constellation economics. Some missions physically can't fit 3 RWs. Prior work exists (AIAA 2023, Li 2013)—you're not alone in thinking this is viable. | Run cost-benefit analysis: $/kg to orbit × mass savings = $X saved per satellite × constellation size. Compare COTS 3RW systems vs 3+1 (volume, mass, power, cost). |
| "Just relax pointing requirements" | Some missions have HARD requirements (optical crosslinks need ~0.1°, Earth observation science). Relaxing goals means mission failure, not mission adaptation. | Survey of mission pointing requirements showing which missions need sub-degree pointing. Show that 3+1 + planner MEETS requirements. |
| "No flight heritage" | BeaverCube-2 using these methods (flight pending). HIL validated on Raspberry Pi. Li 2013 did air-bearing tests—hardware validation exists for the concept. | Document HIL test results with specific metrics. Get BC-2 timeline estimate. |
| "100 Monte Carlo not enough" | 100 runs with varied ICs, orbits, goals is MORE than prior work (single cases). Show convergence plot—results stabilize by ~50 runs. | Generate convergence plot showing metric stability vs run count. Offer to run 1000 if reviewer demands it. |
| "Prior work exists (AIAA 2023)" | They use PD only; you have ALTRO planner. They test single cases; you have Monte Carlo. They don't analyze LTV controllability. They don't address graceful degradation. | Read the paper (in Downloads). Make explicit comparison table. |

### Package Paper Weaknesses

| Weakness | Existing Arguments | Evidence to Gather |
|----------|-------------------|-------------------|
| "Other options exist" | Searched extensively—couldn't find comparable pure-Python closed-loop ADCS packages. Basilisk is C++ core with Python wrappers, message-passing architecture, hard to modify. OpenSatKit is flight software, not ADCS simulation. | Document the search: list packages examined, what they do, why they don't qualify. Make comparison table. Ask community (Reddit r/aerospace, forums) if anyone knows of alternatives. |
| "Python too slow" | HIL tests on Raspberry Pi demonstrate sufficient performance for CubeSat control rates (1-10 Hz). Python is fast enough for development/testing; hot paths can be Cython/C++ if needed. Many CubeSats already fly Python (MicroPython, etc.). | Benchmark: control loop timing on RPi at 1 Hz, 10 Hz. Profile hot paths. Show margin (e.g., "loop runs in 50ms, deadline is 100ms"). |
| "ADCS is small part of mission" | ADCS failures are a leading cause of CubeSat mission failures. Better tools reduce bugs. Rapid architecture trades save months of development. Framework enables trying things that would otherwise be too expensive to prototype. | Cite CubeSat failure statistics (there are papers on this). Show example: "evaluating 3+1 vs 3+0 vs 3+3 took X lines of config changes vs Y weeks of custom code." |
| "No formal verification" | Extensive unit tests, integration tests, Monte Carlo validation. Formal proofs in Generalized Control paper. Practical validation > formal proofs for most small sat teams. | Document test coverage %. Show Monte Carlo validates edge cases. |

### Planner Paper Weaknesses

| Weakness | Existing Arguments | Evidence to Gather |
|----------|-------------------|-------------------|
| "Planners can fail" | Graceful degradation: infeasible goals produce BOUNDED, PREDICTABLE behavior—not crashes, not spinning out. Fallback to PD controller if ALTRO diverges. TVLQR/MPC tracking handles model mismatch online. Two-pass optimization (coarse → fine) improves convergence. | Document failure modes and recovery: "If ALTRO doesn't converge in N iterations, system does X." Show Monte Carlo with intentionally bad ICs—what happens? |
| "Other planners exist" | Compare to eigenaxis + trapezoidal (industry standard): ALTRO handles constraints natively, discovers non-obvious solutions, works for underactuated systems. Those baselines assume 3-axis control and rest-to-rest maneuvers. | Implement eigenaxis baseline, run same scenarios, compare metrics (constraint violations, settling time, fuel cost). |
| "Computational cost" | Two-pass reduces cost. Offline planning + online TVLQR tracking is viable for many missions. Coarse pass: ~100ms on RPi (estimate). Fine pass only when needed. | **MUST GATHER**: Timing benchmarks. Run ALTRO on RPi, measure wall-clock time for coarse/fine passes. Compare to control loop deadline. |
| "Spinning solution is special case" | It's an EXAMPLE demonstrating emergent behavior. The point is the planner finds things humans might miss. Other examples likely exist—the user (you) independently thought of the same solution, validating the planner's "intelligence." | Search for other emergent behaviors in your results. Any case where planner did something unexpected but correct? |

### Generalized Control Paper Weaknesses

| Weakness | Existing Arguments | Evidence to Gather |
|----------|-------------------|-------------------|
| "Lot of math for not much gain" | LP allocation: 17° vs 25.7° error = 33% improvement. That's the difference between meeting and missing a pointing requirement. LTV controllability analysis proves 3+1 works—without it, you're guessing. | Make this concrete: "For mission X with 5° requirement, QP fails 40% of the time, LP fails 10%." |
| "LP vs QP unexplained" | **OPEN ISSUE**—acknowledged. Working hypothesis: unconstrained QP finds minimum-norm solution which may not preserve torque direction; LP preserves direction which matters for Lyapunov stability. | **MUST RESOLVE**: Either (a) prove LP direction preservation matters theoretically, (b) find QP constraints that match LP, or (c) acknowledge as empirical finding with hypothesis. |
| "Stability proofs missing" | Direction preservation maintains Lyapunov descent condition (if controller assumes τ direction, and allocation preserves it, stability follows). | Add Lyapunov analysis showing: if V̇ < 0 for desired τ, and LP produces τ' with same direction, then V̇ < 0 for τ'. |
| "QPC <6% means constraints don't matter" | Low activation in nominal operation is GOOD design—system doesn't need emergency constraints. But when you DO need them (edge cases, failures), they prevent catastrophe. Like a seatbelt—rarely used, essential when needed. | Run scenarios that DO activate QPC. Show what happens without constraints in those cases (bad) vs with (bounded). |
| "LTV assumes known B-field" | IGRF model is well-characterized (errors < 1% in LEO). Estimator can adapt to measured field. Even with 10% B-field error, LTV analysis still provides useful bounds. | Sensitivity analysis: how much does pointing degrade with X% B-field error? |

---

## Evidence Generation Priority List

**Must-have for conference submissions:**
1. [ ] Read AIAA 2023 paper, make comparison table
2. [ ] Timing benchmarks for ALTRO on RPi
3. [ ] Document Python package search (what you looked at, why it doesn't qualify)
4. [ ] 5-minute demo case study for Package paper
5. [ ] Convergence plot for Monte Carlo (show results stabilize)

**Must-have for journal submissions:**
1. [ ] LP vs QP theoretical explanation or acknowledgment
2. [ ] Eigenaxis baseline comparison for Planner paper
3. [ ] Stability proof sketch for allocation layer
4. [ ] Break-even analysis for 3+1 vs 3+3
5. [ ] Formal Basilisk comparison

**Nice-to-have:**
1. [ ] Cost-benefit analysis ($/kg × mass savings × constellation size)
2. [ ] CubeSat failure statistics citation
3. [ ] More Monte Carlo runs (1000) if reviewers demand
4. [ ] Additional emergent planner behaviors

---

## Python ADCS Package Comparison (Package Paper Support)

### Packages Found and Analyzed

| Package | Language | Closed-Loop ADCS? | Actuators | Maintenance | Limitations |
|---------|----------|-------------------|-----------|-------------|-------------|
| **Basilisk** | C++ (Python wrapper) | ✅ Yes | Many | ✅ Active (CU Boulder/LASP) | Message-passing architecture, hard to modify, C++ core |
| **42** (NASA Goddard) | C (74%) | ✅ Yes | Many | ⚠️ Informal | No native Python, socket IPC only |
| **adcs-simulation** (gavincmartin) | Python | ⚠️ Basic | Unknown | ❌ Dormant (course project) | "No releases published," limited docs |
| **AWP** (alfonsogonzalez) | Python | ⚠️ Partial | Unknown | ⚠️ Moderate | Educational focus, attitude videos exist but unclear if closed-loop |
| **attitude_control_reaction_wheels** | Python | ✅ Yes (RW only) | 4 RW pyramid | ❌ Small project | Single config only, no MTQ, no estimation |
| **risherlock/adcs** | C/embedded | ❌ Utilities only | N/A | Unknown | Targeted at embedded, not simulation |
| **ADCS_demo** (Open Cosmos) | Python | ⚠️ Demo only | MTQ | ❌ Demo | "Demonstration of principle," not full sim |
| **PyCubed ADCS** | Python | ✅ Yes | MTQ | ⚠️ Mission-specific | Tied to PyCubed hardware platform |

### Your Package's Unique Position

**What none of them have (that yours does)**:
1. ✅ Pure Python with full closed-loop simulation
2. ✅ Multiple actuator types (MTQ, RW, thrusters) in unified framework
3. ✅ Configurable allocation layer (LP, QP, etc.)
4. ✅ Full- and reduced-attitude goals
5. ✅ Integrated estimation (UAKF/SRUAKF)
6. ✅ Trajectory planning (ALTRO)
7. ✅ HIL testing support
8. ✅ Designed for flight software (not just simulation)
9. ✅ Disturbance modeling and feedforward

**Basilisk comparison specifically**:
- Basilisk: C++ core with Python scripting → fast but hard to modify internals
- Yours: Pure Python → slower but highly modular, easy to extend, lower barrier to entry
- Basilisk: Message-passing architecture → complex data flow
- Yours: Direct function calls → clearer control flow, easier debugging

**Suggested framing for Package paper**:
> "We surveyed existing open-source ADCS tools and found no pure-Python package offering integrated closed-loop simulation with configurable estimation, control, allocation, and planning. Basilisk [cite] provides comprehensive simulation but uses a C++ core with message-passing architecture that is difficult to modify for rapid prototyping. Our framework fills this gap by providing a modular, pure-Python solution optimized for development speed and accessibility while maintaining flight-readiness through HIL validation."

Sources:
- [Basilisk](https://hanspeterschaub.info/basilisk/)
- [42 Simulation](https://github.com/ericstoneking/42)
- [adcs-simulation](https://github.com/gavincmartin/adcs-simulation)
- [AWP](https://github.com/alfonsogonzalez/AWP)

---

## ADCS Importance: Failure Statistics (For "ADCS is small part of mission" argument)

### CubeSat Failure Statistics

**Overall reliability**:
- ~25% of CubeSats fail entirely
- Only 54% still operational after 2 years
- Less than half achieve full mission objectives

**Subsystem failure breakdown** (from Langer et al., SSC16):
- **EPS (Electrical Power)**: Largest contributor overall, 14%+ of failures, >40% of failures after 30 days
- **Communications**: ~30% of failures after 90 days; 1/3 of failed missions never contacted after launch
- **ADCS**: Identified as "leading contributor to failures in small satellite missions" alongside EPS and Comms

**Key quotes for your paper**:
> "Most CubeSat failures originate in the Electrical Power System (EPS), Attitude Determination and Control System (ADCS), and the communications system. These subsystems are mission-critical; if any of these subsystems fail, the entire satellite experiences failure."

> "The ADCS plays a pivotal role in managing the satellite's orientation in space. However, this importance highlights a concerning issue: the ADCS has been identified as a leading contributor to failures in small satellite missions."

> "The analysis of these failures shows that primarily insufficient functional testing on system level led to low success rates in former missions."

**Why ADCS matters (argument for your Package paper)**:
1. ADCS is one of the top 3 failure sources
2. ADCS failures = total mission loss (can't point = can't do anything)
3. Better ADCS tools → better testing → fewer failures
4. CubeSat reliability increases more through improved testing than redundancy
5. Your framework enables rapid testing across configurations

**Counter-argument to "ADCS is small part of mission"**:
> "While ADCS is a single subsystem, it is mission-critical: a satellite that cannot point cannot communicate, generate power (if sun-tracking), or accomplish science objectives. ADCS failures are among the top three contributors to CubeSat mission failure, alongside EPS and communications. Better ADCS development tools directly address the root cause identified in failure analyses: insufficient system-level testing."

Sources:
- [Langer et al. SSC16](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=3397&context=smallsat)
- [NASA S3VI ADCS Knowledge Base](https://s3vi.ndc.nasa.gov/ssri-kb/topics/48/)
- [CubeSat reliability study](https://www.sciencedirect.com/science/article/pii/S0951832021007584)

---

## ALTRO as Pilot Analogy (For Planner Paper Narrative)

### Traditional Control (PD/LQR) = Autopilot

Like a car's cruise control or aircraft autopilot:
- **Pre-set parameters**: You input heading, altitude, speed
- **Reactive corrections**: Adjusts control surfaces to maintain inputs
- **Cannot handle unexpected situations**: In turbulence or emergencies, human takes over
- **No decision-making**: "A very smart calculator" - follows rules, doesn't reason

**Quote**: "Autopilot is not an AI—it's more like a very smart calculator. It relies entirely on pre-set parameters and data inputs."

### ALTRO/Trajectory Planning = Skilled Pilot

Like a trained pilot navigating complex airspace:
- **Situation awareness**: Understands current state, goals, constraints
- **Route planning**: Chooses path considering obstacles, weather, fuel
- **Adaptation**: Adjusts plan when conditions change
- **Creative problem-solving**: Finds non-obvious solutions (like your spinning-for-disturbance-rejection)
- **Graceful handling of problems**: When destination unreachable, finds best alternative

**Your ALTRO advantages using this analogy**:

| Capability | PD Controller (Autopilot) | ALTRO Planner (Pilot) |
|------------|---------------------------|------------------------|
| Following a route | ✅ Excellent | ✅ Excellent |
| Obstacle avoidance | ❌ Needs external input | ✅ Native constraint handling |
| Handling emergencies | ❌ Falls back to human | ✅ Graceful degradation |
| Creative solutions | ❌ None | ✅ Discovers spinning strategy |
| Adapting to unknowns | ⚠️ Limited (integral term) | ✅ Replanning capability |
| "What if goal is impossible?" | ❌ Oscillates/saturates | ✅ Converges to closest achievable |

### Specific Analogy for Paper

> "Traditional attitude controllers like PD operate as autopilots: they track a commanded trajectory but cannot reason about whether that trajectory is achievable or optimal. When goals become infeasible—due to actuator limits, disturbances, or geometric constraints—these controllers may saturate actuators, accumulate error, or oscillate indefinitely.
>
> ALTRO-based trajectory planning functions more like a skilled pilot. Given a destination and constraints, it plans a feasible route, adapts when conditions change, and—critically—recognizes when the destination is unreachable and converges to the best achievable alternative. The spinning solution discovered for overwhelming disturbances (Section X) exemplifies this: the planner found a creative strategy that a simple controller would never attempt, analogous to a pilot choosing an unconventional approach when standard procedures fail."

### MIT Research on Advanced Planning (supporting evidence)

MIT researchers developed algorithms for "stabilize-avoid" scenarios:
> "Navigating extreme scenarios that a human wouldn't be able to handle is where their approach really shines."

This is exactly what ALTRO does for spacecraft—handles scenarios that simple controllers can't.

Sources:
- [Autopilot vs Pilot comparison](https://avi-8.com/blogs/the-aviation-journal/how-autopilot-really-works-myths-vs-reality)
- [MIT safe autopilots research](https://news.mit.edu/2023/safe-and-reliable-autopilots-flying-0612)
- [ALTRO-C for MPC](https://rexlab.ri.cmu.edu/papers/ALTRO_MPC.pdf)

---

## Simulation Tools: NASA 42 vs Basilisk

### NASA 42

**Overview**: 42 is a comprehensive general-purpose simulation of spacecraft attitude and orbit dynamics developed by Eric Stoneking at NASA Goddard Space Flight Center. Named as a reference to "The Hitchhiker's Guide to the Galaxy" (mostly harmless).

**Key Features**:
- High-fidelity multi-body spacecraft attitude dynamics (rigid and/or flexible bodies)
- Both two-body and three-body orbital flight regimes
- Environments from LEO throughout the solar system
- Multiple spacecraft simulation for formation flying
- Built-in 3D visualization
- Open source since 2014

**Architecture**:
- Core written in C (~74%)
- Configuration via text files
- Socket-based inter-process communication for external controllers
- Standalone executable

**When to use 42**:
- ✅ Full mission simulation with high fidelity
- ✅ Multi-spacecraft scenarios
- ✅ Visualizing attitude and orbits
- ❌ Not easily extensible (C codebase)
- ❌ No native Python integration
- ❌ Requires socket IPC for external control

**Resources**:
- [GitHub Repository](https://github.com/ericstoneking/42)
- [NASA Technical Report](https://ntrs.nasa.gov/citations/20180000954)

---

### Basilisk

**Overview**: Basilisk is an open-source astrodynamics simulation framework developed by the Autonomous Vehicle Systems Lab at CU Boulder and LASP. Designed for both research and mission development.

**Key Features**:
- Modular architecture with strict decoupling of modeling concerns
- Faster-than-realtime and repeatable Monte-Carlo simulations
- Real-time hardware-in-the-loop capability
- Accompanying Vizard visualization tool (Unity-based)
- Cross-platform (macOS, Windows, Linux)

**Architecture**:
- **C++ core** for execution speed
- **Python wrapper** for scripting and configuration
- **Message-passing** between modules (not direct function calls)
- Modules, Tasks, and Task Groups as core components

**When to use Basilisk**:
- ✅ High-fidelity Monte Carlo campaigns
- ✅ Hardware-in-the-loop testing
- ✅ Standard spacecraft configurations
- ⚠️ Harder to modify internal algorithms
- ⚠️ Message-passing architecture adds complexity
- ⚠️ Steeper learning curve

**Key Difference from Your Package**:
| Aspect | Basilisk | Your Package |
|--------|----------|--------------|
| Core language | C++ | Pure Python |
| Scripting | Python wrapper | Native Python |
| Architecture | Message-passing | Direct function calls |
| Modifiability | Hard (C++ core) | Easy (Python) |
| Speed | Very fast | Slower but sufficient for HIL |
| Target users | Advanced users | Rapid prototyping, students, small teams |

**Resources**:
- [Official Documentation](https://avslab.github.io/basilisk/)
- [GitHub Repository](https://github.com/AVSLab/basilisk)
- [AIAA Paper](https://arc.aiaa.org/doi/10.2514/1.I010762)

---

## COMPREHENSIVE LITERATURE REVIEW

### Paper 1: 3+1 Architecture (3 MTQ + 1 RW Control)

#### Key Background Papers (10-20 Must-Review)

1. **Wie & Barba (1985)** - "Quaternion Feedback for Spacecraft Large Angle Maneuvers" - *JGCD Vol. 8, pp. 360-365*
   - Foundational paper on quaternion PD control
   - Lyapunov stability proofs for 3-axis maneuvers
   - [ResearchGate](https://www.researchgate.net/publication/286349928_Quaternion_feedback_for_spacecraft_large_angle_maneuvers)

2. **Silani & Lovera (2005)** - "Magnetic Spacecraft Attitude Control: A Survey and Some New Results" - *Control Engineering Practice Vol. 13, pp. 357-371*
   - Comprehensive survey of magnetic control approaches
   - Periodic control theory for LTV systems
   - Model predictive control for magnetic actuation
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0967066103002922)

3. **Wisniewski & Blanke (1999)** - "Fully Magnetic Attitude Control for Spacecraft Subject to Gravity Gradient" - *Automatica Vol. 35, No. 7, pp. 1201-1214*
   - Passivity-based approach for magnetic-only control
   - Exploits periodic nature of Earth's magnetic field
   - Seminal work on LTV controllability for magnetic systems

4. **Lovera et al. (2002)** - "Periodic Attitude Control Techniques for Small Satellites with Magnetic Actuators" - *IEEE Trans. Control Systems Technology Vol. 10, No. 1, pp. 90-95*
   - Optimal discrete-time magnetic control design
   - Periodic LQR approach
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/974341/)

5. **Zhu et al. (2023)** - "Attitude Control of a 3U CubeSat with Combination of Magnetorquers and Reaction Wheels" - *AIAA SciTech Forum*
   - **MUST CITE** - Direct prior work on 3+1 architecture
   - PD controller, single-case testing
   - [AIAA](https://arc.aiaa.org/doi/10.2514/6.2023-0935)

6. **Li et al. (2013)** - "Design of Attitude Control Systems for CubeSat-Class Nanosatellite" - *Journal of Control Science and Engineering*
   - Sliding-mode + fuzzy logic control
   - Air-bearing hardware validation
   - [Hindawi](https://www.hindawi.com/journals/jcse/2013/657182/)

7. **Forbes & Damaren (2020)** - "Hybrid Attitude Control for Nano-Spacecraft: Reaction Wheel Failure and Singularity Handling" - *JGCD*
   - RW failure modes and MTQ backup
   - Singularity avoidance in allocation
   - [AIAA](https://arc.aiaa.org/doi/10.2514/1.G005525)

8. **Tregouet et al. (2015)** - "Reaction Wheels Desaturation Using Magnetorquers and Static Input Allocation" - *IEEE Trans. Control Systems Technology*
   - Allocation-based momentum management
   - Decoupling attitude control from desaturation
   - [HAL](https://hal.science/hal-01760720v1/file/Attitude-Allocation.pdf)

9. **Avanzini & Giulietti (2012)** - "Magnetic Detumbling of a Rigid Spacecraft" - *JGCD Vol. 35, No. 4*
   - B-dot law analysis and alternatives
   - Stability proofs for magnetic control

10. **Sidi (1997)** - "Spacecraft Dynamics and Control: A Practical Engineering Approach" (Book)
    - Standard textbook for attitude dynamics
    - Disturbance torque modeling
    - Actuator sizing

11. **Bhat (2005)** - "Controllability of Nonlinear Time-Varying Systems: Applications to Spacecraft Attitude Control Using Magnetic Actuation" - *IEEE Trans. Automatic Control Vol. 50, No. 11, pp. 1725-1735*
    - Theoretical foundation for LTV controllability
    - Necessary conditions for magnetic control

12. **Psiaki (2001)** - "Magnetic Torquer Attitude Control via Asymptotic Periodic Linear Quadratic Regulation" - *JGCD Vol. 24, No. 2*
    - Optimal magnetic control design
    - Periodic LQR formulation

#### Must-Read-Fully (1-4 papers)

1. **⭐ Zhu et al. (2023)** - AIAA SciTech - *You must differentiate from this!*
2. **⭐ Silani & Lovera (2005)** - The definitive survey on magnetic control
3. **⭐ Wie & Barba (1985)** - Foundational quaternion control
4. **Forbes & Damaren (2020)** - RW failure and hybrid architectures

---

### Paper 2: Python ADCS Package

#### Key Background Papers (10-20 Must-Review)

1. **Markley & Crassidis (2014)** - "Fundamentals of Spacecraft Attitude Determination and Control" (Book)
   - **THE textbook** for ADCS
   - Estimation, control, sensors, actuators
   - [Springer](https://link.springer.com/book/10.1007/978-1-4939-0802-8)

2. **Crassidis et al. (2007)** - "Survey of Nonlinear Attitude Estimation Methods" - *JGCD Vol. 30, No. 1, pp. 12-28*
   - Comprehensive estimation survey
   - EKF, UKF, particle filters
   - [Buffalo PDF](https://ancs.eng.buffalo.edu/pdf/ancs_papers/2007/att_survey07.pdf)

3. **Crassidis & Markley (2003)** - "Unscented Filtering for Spacecraft Attitude Estimation" - *JGCD Vol. 26, No. 4, pp. 536-542*
   - Original UAKF for attitude
   - Quaternion estimation challenges

4. **Markley (2004)** - "Attitude Estimation or Quaternion Estimation?" - *J. Astronautical Sciences Vol. 52*
   - Critical distinction for filter design
   - Three-parameter vs quaternion estimation
   - [Springer](https://link.springer.com/article/10.1007/BF03546430)

5. **Kenneally et al. (2018)** - "Basilisk: A Flexible, Scalable and Modular Astrodynamics Simulation Framework" - *AIAA JAIS*
   - Primary comparison target
   - Architecture and design philosophy
   - [AIAA](https://arc.aiaa.org/doi/10.2514/1.I010762)

6. **Stoneking (2018)** - "42: An Open-Source Simulation Tool for Study and Design of Spacecraft Attitude Control Systems" - *NASA Technical Report*
   - NASA's simulation philosophy
   - [NTRS](https://ntrs.nasa.gov/citations/20180000954)

7. **Langer et al. (2016)** - "Reliability of CubeSats – Statistical Data, Developers' Beliefs and the Way Forward" - *SSC16*
   - CubeSat failure statistics
   - ADCS as failure source
   - [USU](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=3397&context=smallsat)

8. **Shuster (1993)** - "A Survey of Attitude Representations" - *J. Astronautical Sciences Vol. 41, No. 4*
   - Quaternion and rotation representations
   - Essential for any ADCS implementation

9. **Wertz (1978/2012)** - "Spacecraft Attitude Determination and Control" (Book)
   - Classic reference text
   - Mission-oriented approach

10. **Open-Source Projects to Acknowledge**:
    - poliastro (Python astrodynamics)
    - OREKIT (Java/Python orbit mechanics)
    - SatNOGS (ground station software)

#### Must-Read-Fully (1-4 papers)

1. **⭐ Markley & Crassidis (2014)** - The textbook; establishes conventions
2. **⭐ Crassidis et al. (2007)** - Estimation survey; your UAKF context
3. **⭐ Kenneally et al. (2018)** - Basilisk paper; primary differentiator
4. **Langer et al. (2016)** - Failure statistics for "why ADCS matters"

---

### Paper 3: Generalized Control (LTI/LTV Allocation)

#### Key Background Papers (10-20 Must-Review)

1. **Wie & Barba (1985)** - Quaternion PD control foundation (same as 3+1)

2. **Silani & Lovera (2005)** - Magnetic control survey with LTV analysis

3. **Bodson (2002)** - "Evaluation of Optimization Methods for Control Allocation" - *JGCD Vol. 25, No. 4*
   - LP vs QP for allocation
   - Direction preservation discussion

4. **Johansen & Fossen (2013)** - "Control Allocation—A Survey" - *Automatica Vol. 49, pp. 1087-1103*
   - Comprehensive allocation methods
   - Spacecraft applications
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0005109813000654)

5. **Oppenheimer et al. (2006)** - "Control Allocation for Over-Actuated Systems" - *IEEE Control Systems Magazine*
   - When you have more actuators than DOF
   - Optimization approaches

6. **Durham (1993)** - "Constrained Control Allocation" - *JGCD Vol. 16, No. 4*
   - Early work on constrained allocation
   - Geometric approaches

7. **Kim et al. (2025)** - "Spacecraft Attitude Control with On-Off Thrusters via Convex Optimization Based Control Allocation" - *Int. J. Aeronautical and Space Sciences*
   - Recent work on LP/QP for spacecraft
   - [Springer](https://link.springer.com/article/10.1007/s42405-025-00904-y)

8. **Bhat (2005)** - LTV controllability (same as 3+1 list)

9. **Lovera & Astolfi (2006)** - "Global Magnetic Attitude Control of Spacecraft in the Presence of Gravity Gradient" - *IEEE Trans. Aerospace and Electronic Systems Vol. 42, No. 3*
   - Combined magnetic and gravity gradient
   - Stability analysis

10. **Pulecchi et al. (2010)** - "Optimal Discrete-Time Design of Three-Axis Magnetic Attitude Control Laws" - *IEEE Trans. Control Systems Technology Vol. 18, No. 3*
    - Optimal magnetic control design

11. **Petersen et al. (2000)** - "Robust Control Design Using H-infinity Methods" (Book)
    - Robust control background for LTV systems

12. **Zhou & Doyle (1998)** - "Essentials of Robust Control" (Book)
    - Linear systems theory background

#### Must-Read-Fully (1-4 papers)

1. **⭐ Johansen & Fossen (2013)** - THE allocation survey paper
2. **⭐ Bodson (2002)** - LP vs QP evaluation (directly relevant to your LP mystery!)
3. **⭐ Silani & Lovera (2005)** - LTV magnetic control theory
4. **Lovera & Astolfi (2006)** - Global stability for magnetic control

---

### Paper 4: MTQ Trajectory Planner (ALTRO)

#### Key Background Papers (10-20 Must-Review)

1. **Howell, Jackson & Manchester (2019)** - "ALTRO: A Fast Solver for Constrained Trajectory Optimization" - *IROS 2019*
   - **THE ALTRO paper**
   - Augmented Lagrangian + iLQR
   - [PDF](https://bjack205.github.io/assets/ALTRO.pdf)

2. **Li & Todorov (2004)** - "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems" - *ICINCO*
   - Original iLQR formulation

3. **Todorov & Li (2005)** - "A Generalized Iterative LQG Method for Locally-Optimal Feedback Control of Constrained Nonlinear Stochastic Systems" - *ACC 2005*
   - iLQG extensions

4. **Malyuta et al. (2022)** - "Advances in Trajectory Optimization for Space Vehicle Control" - *Annual Reviews in Control*
   - Comprehensive survey of space trajectory optimization
   - Convexification methods
   - [NSF](https://par.nsf.gov/servlets/purl/10354182)

5. **Boyarko et al. (2011)** - "Optimal Feedback Control for the Stationary Axis Large Angle Slew Maneuver" - *JGCD Vol. 34, No. 6*
   - Eigenaxis alternatives
   - Optimal slew planning

6. **McInnes (1994)** - "Large Angle Slew Maneuvers with Autonomous Sun Vector Avoidance" - *JGCD*
   - Keep-out constraint handling
   - Geometric approaches

7. **Hablani (1994)** - "Sun-Tracking Commands and Reaction Wheel Sizing with Configuration Optimization" - *JGCD Vol. 17, No. 4*
   - Constraint-aware planning

8. **Kluever (2012)** - "Sun-Avoidance Slew Planning with Keep-Out Cone and Actuator Constraints" - *J. Spacecraft and Rockets*
   - Pontryagin-based planning
   - [AIAA](https://arc.aiaa.org/doi/10.2514/1.A34671)

9. **Jackson et al. (2021)** - "AL-iLQR Tutorial"
   - Augmented Lagrangian tutorial
   - [PDF](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)

10. **Nocedal & Wright (2006)** - "Numerical Optimization" (Book)
    - Optimization theory background
    - Augmented Lagrangian methods

11. **CMU RExLab (2019)** - "Magnetorquer-Only Attitude Control of Small Satellites"
    - Trajectory optimization for magnetic control
    - [PDF](https://rexlab.ri.cmu.edu/papers/magnetorquer_only.pdf)

12. **Bryson & Ho (1975)** - "Applied Optimal Control" (Book)
    - Classic optimal control theory

13. **Bemporad et al. (2015)** - "Constrained Model Predictive Control of Spacecraft Attitude"
    - MPC for attitude with constraints
    - [PDF](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ecc15-reaction-wheels.pdf)

14. **Aguilar-Ibanez et al. (2020)** - "Attitude Control of Low Earth Orbit Satellites by Reaction Wheels and Magnetic Torquers"
    - Combined RW + MTQ control
    - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S009457651831974X)

#### Must-Read-Fully (1-4 papers)

1. **⭐ Howell et al. (2019)** - THE ALTRO paper; understand your solver
2. **⭐ Malyuta et al. (2022)** - Comprehensive trajectory optimization survey
3. **⭐ Jackson et al. (2021)** - AL-iLQR tutorial for understanding internals
4. **CMU RExLab (2019)** - Magnetorquer trajectory optimization (closest prior work to yours)

---

## Cross-Cutting References (All Papers)

### Textbooks

1. **Markley & Crassidis (2014)** - "Fundamentals of Spacecraft Attitude Determination and Control"
2. **Sidi (1997)** - "Spacecraft Dynamics and Control"
3. **Wertz (1978/2012)** - "Spacecraft Attitude Determination and Control"
4. **Nocedal & Wright (2006)** - "Numerical Optimization"

### Survey Papers

1. **Crassidis et al. (2007)** - Attitude estimation survey
2. **Silani & Lovera (2005)** - Magnetic control survey
3. **Johansen & Fossen (2013)** - Control allocation survey
4. **Malyuta et al. (2022)** - Trajectory optimization survey

### CubeSat/Small Satellite Statistics

1. **Langer et al. (2016)** - CubeSat reliability study
2. **Swartwout (2013)** - CubeSat database and statistics

---

## Quick Reference: BibTeX Keys (Suggested)

```bibtex
% Foundational
@article{wie1985quaternion, ...}
@book{markley2014fundamentals, ...}
@article{silani2005magnetic, ...}

% Estimation
@article{crassidis2007survey, ...}
@article{crassidis2003unscented, ...}

% Prior 3+1 Work
@inproceedings{zhu2023attitude, ...}
@article{li2013design, ...}
@article{forbes2020hybrid, ...}

% Allocation
@article{johansen2013control, ...}
@article{bodson2002evaluation, ...}

% Trajectory Planning
@inproceedings{howell2019altro, ...}
@article{malyuta2022advances, ...}

% Simulation Tools
@article{kenneally2018basilisk, ...}
@techreport{stoneking201842, ...}

% Statistics
@inproceedings{langer2016reliability, ...}
```

---

## INTERVIEW CONTINUATION (Session 2)

### KEY INSIGHT: CONOPS Coupling as Unifying Theme

**Patrick's insight**: ADCS should be coupled with mission CONOPS from the start, not treated as an independent subsystem.

**What makes ADCS special vs other subsystems?**
- Other subsystems set strict requirements (e.g., "always maintain 1° pointing")
- ADCS can *change what you can do* — reframe goals dynamically
- Example: Instead of "maintain 1° pointing always," you could say "maintain 1° around optical axis with some maximum spin during imaging times only"
- This flexibility enables mission creativity, not just requirement satisfaction

**Proposed thesis statement for all papers:**
> "ADCS design should be coupled with mission CONOPS from the start, not treated as an independent subsystem. This work provides the tools—estimation, control, allocation, and planning—that enable rapid evaluation of ADCS architectures against mission requirements, revealing which configurations are viable before hardware is selected."

**How each paper connects to CONOPS:**

| Paper | CONOPS Connection |
|-------|-------------------|
| **3+1** | "For missions with profile X (nadir pointing, moderate slew rates, LEO), 3+1 is the optimal CONOPS choice" — proves a cheaper architecture meets requirements for certain profiles |
| **Package** | Makes CONOPS-coupled ADCS design accessible; enables rapid exploration of architecture-mission trade space; reduces dev time |
| **Planner** | Adapts trajectories to CONOPS constraints; makes underactuated satellites do more; finds novel solutions operators wouldn't think of |
| **Generalized Control** | Provides modular math that makes architecture comparisons tractable (less direct mission enablement, more foundational) |

**Mission impact framing:**
- **Big missions**: Made cheaper (same capability, less hardware)
- **Small missions**: Made more capable (better performance from limited hardware)
- **New missions**: Made possible (previously infeasible due to mass/power/cost constraints)

---

### Paper Status Update (Jan 2026)

| Paper | Status | Conference | Deadline Notes |
|-------|--------|------------|----------------|
| 3+1 | Abstract only, writing + some sims needed | SmallSat Europe (May 2026) | ~4 months |
| Package | Structure done, [TBD]s remain, sims + examples needed | SmallSat Europe (May 2026) | ~4 months |
| Planner | Most complete | SmallSat USA | Later |
| Generalized Control | Math done (Niclas), narrative needs work | SmallSat USA | Later |

**Patrick's confidence level**: Not worried about timeline. Package is robust, they've written many simulations recently. Abstracts not even accepted yet.

---

### LP vs QP Update

**Status**: Still working on it.

**Patrick's approach**: "The QP solution should include the LP, so if the right constraint was added it should always perform at least as well as LP. Working on that constraint. Will present alternate options in paper and highlight which is best."

**Plan for paper**: Present multiple allocation options with empirical comparison, explain the constraint that makes QP competitive.

---

### Generalized Control Paper - Narrative Gap

**Origin story** (compelling): Patrick's advisor wanted comparisons to other control laws. Each comparison was hard because different laws assumed different actuators, orbits, goals, disturbances. Patrick wanted a modular "bolt-on" front-end and back-end that could wrap existing laws for rapid testing.

**Problem**: Current paper content (LTV controllability, LP vs QP allocation) doesn't clearly deliver on the "bolt-on adapter" promise. Reads more like pure theory than practical tool.

**Action needed**: More writing to connect the math to the "rapid architecture testing" motivation. Niclas did the math; Patrick needs to add the framing.

---

### Graceful Degradation - Refined Definition

> **Graceful degradation** means that when the planner receives an infeasible goal, it:
> 1. Does not error out or return no solution
> 2. Does not command actuators to saturation indefinitely ("doesn't do anything insane")
> 3. Converges to the closest achievable attitude that minimizes goal error while respecting actuator and rate constraints ("tries its best to meet goals")
> 4. Always meets constraints (rate limits, actuator limits, keep-out zones)
> 5. Produces a bounded, predictable trajectory that can be tracked by the feedback controller
>
> In implementation, if ALTRO does not converge within the pre-computation window, the system falls back to PD control with the current best trajectory estimate.

---

### 3+1 Mission Examples (To Be Validated with Numbers)

**Option A: 1U Imaging CubeSat**
- University team wants 1U Earth-observation with sub-degree pointing
- 3RW won't fit alongside camera + OBC
- MTQ-only gives ~5-10°, insufficient
- 3+1 fits and achieves <1° with planner
- Claim: "3+1 enables imaging missions at 1U scale previously impossible"

**Option B: Constellation Economics**
- 200-satellite LEO constellation for IoT
- 3RW × 200 = $3M in RWs (at $5k/wheel)
- Mass: 200g × 200 = 40kg extra to orbit (~$400k at $10k/kg)
- 3+1 saves ~$2M+ across constellation
- Claim: "Significant cost savings for constellation missions"

**Option C: Power-Constrained Eclipse Mission**
- 3U studying aurora, long eclipse periods
- 3RW draws ~1.5W continuous; 3+1 saves ~1W
- 40-min eclipse: saves 0.67 Wh per eclipse
- Claim: "40% power reduction enables extended eclipse operations"

**Option D: BeaverCube-2 (Real Example)**
- BC-2 used 3+1, it "became more feasible"
- Need to document: what constraint was binding (mass? power? volume? cost?)
- What pointing accuracy was required?
- What would have been sacrificed for 3RW?

---

### Why ADCS is Special for CONOPS Coupling (Evolving Argument)

**The core insight**: Unlike other subsystems, ADCS has *discrete capability thresholds* that depend on mission profile.

**Other subsystems scale linearly:**
- More solar panels → more power (proportional)
- Bigger antenna → more bandwidth (proportional)
- Faster processor → more compute (proportional)

**ADCS has discrete capability jumps:**
- 3MTQ → limited goal types, time-varying constraints, can't guarantee arbitrary pointing
- 3MTQ + 1RW → *qualitatively different*: new goals achievable, but constraints remain orbit-dependent
- 3RW → full authority, but potentially overkill

**Key insight**: Adding one RW doesn't give "33% more pointing"—it *unlocks an entirely new capability class*. Whether that class is sufficient depends on specific CONOPS (when do you need pointing? what goals? what orbit?).

**Why this is non-obvious:**
1. Mission designers think in requirements ("always maintain 1°") not capabilities ("can achieve 1° during imaging windows with setup time")
2. Capability boundaries depend on orbit, timing, goal type—not just hardware specs
3. Standard tools don't reveal these boundaries without proper analysis

**Patrick's additional thoughts:**
- Designers already allow operational modes for power (different modes, time-sharing); why not ADCS?
- Could reframe pointing as "meet 1° during imaging times" rather than "always maintain 1°"
- ADCS is directly coupled to mission success—it determines whether payload can do its job (pointing), whereas power/thermal/compute/radio are about keeping systems alive and moving data

**Proposed framing (draft, may vary per paper):**
> "Unlike power or communications—where more hardware provides proportionally more capability—ADCS exhibits discrete capability thresholds that depend on mission profile. Adding a single reaction wheel to a magnetorquer-only system doesn't improve pointing by 33%; it unlocks an entirely new class of achievable missions. But which missions fall within that class depends on orbit, timing, and goal formulation—factors invisible without proper analysis tools. This work provides those tools."

**STATUS**: Close but Patrick feels it's still missing something. Will vary per paper. Needs further iteration.

---

### Final Interview Questions (Session 2)

**LP vs QP Resolution Plan:**
- If constraint isn't found: include both results, demonstrate LP as better empirically
- Paper WILL be more about modular framework than currently written
- LP vs QP becomes supporting evidence, not central claim

**Data Consistency Plan:**
- Will probably rerun simulations
- Different scenarios may warrant both sets of numbers
- Baseline requirement: internal consistency within each paper

**Experiment Concerns:**
- NOT worried about: Generalized Control, Package, 3+1
- MAYBE worried about: Some Planner re-runs
- Paper refinement process helps identify what's vital vs what's not

---

### Generalized Control Paper - Framing Decision

**Three possible actionable takeaways:**

| Option | Takeaway | Audience | Action After Reading |
|--------|----------|----------|---------------------|
| **A: LP Allocator** | "Use LP instead of QP for underactuated allocation—preserves direction, improves stability" | Engineers implementing flight software | Implement LP allocation |
| **B: LTV Controllability** | "Use this analysis to verify your underactuated config is viable before committing" | Systems engineers doing trades | Run controllability checks |
| **C: Modular Framework** | "This is a 'bolt-on adapter' for any control law—swap in PD, LQR, sliding mode, we handle actuator details" | Researchers, architecture teams | Rapid control law comparison |

**Patrick's decision:**
- Wants **A + C** (LP allocator + modular framework for rapid testing)
- **B** (LTV controllability) serves as supporting proof that combinations work
- More math than API (Package paper will have the practical tool)
- Niclas (grad student) may want B emphasis
- **More writing needed** to deliver on C framing—current paper is heavy on math (A, B) but light on modular framework narrative (C)

**Clean split possibility:**
- Generalized Control (JGCD): Theoretical math (A + B proofs) + framework concept (C)
- Package Paper (SmallSat/JOSS): Practical implementation of C with accessible API

---

### 3+1 Mission Examples - VALIDATED NUMBERS

#### Cost Data (2024-2025)

**Reaction Wheel Costs:**
- Commercial CubeSat RWs: **$5,000-$10,000 each** (typical)
- Higher-performance RWs: $20,000-$100,000+
- Research/COTS approach: ~$330 for 3-axis (HDD-based, not flight-grade)
- Sources: [TY-Space](https://www.ty-space.net/understanding-reaction-wheel-price-factors-and-cost-considerations-for-satellites/), [SatSearch](https://satsearch.co/products/kongsberg-nanoavionics-cubesat-reaction-wheel)

**Launch Costs (SpaceX Rideshare):**
- Current pricing: **$5,500-$6,000/kg** to SSO
- Base package: $1.1M per 200 kg
- Compared to dedicated small launcher: 5-10× more expensive
- Sources: [SatBase](https://satbase.com/articles/cubesat-launch-costs), [SpaceX Rideshare](https://www.spacex.com/rideshare)

#### 1U CubeSat Constraints

- Dimensions: **10×10×10 cm**
- Mass limit: **1.33 kg**
- Power constraint: **~40% of total 1U power** required just for RW slew maneuvers
- Volume: Limited accommodation; RW222 designed for 1-3U, RW400 for 6-12U
- Reliability concern: **~50% of ADCS failures** attributed to RW moving parts
- Sources: [Cal Poly Thesis](https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=3705&context=theses), [AAC Clyde Space](https://www.aac-clyde.space/what-we-do/space-products-components/adcs/rw222)

---

#### VALIDATED EXAMPLE A: 1U Imaging CubeSat

**Scenario**: University team building 1U Earth-observation CubeSat with sub-degree pointing.

**The Problem**:
- 1U volume = 10×10×10 cm total
- Camera + OBC + power + comms already consume most volume
- 3 reaction wheels (even small RW222 at 50×50×27mm each) consume significant fraction of remaining space
- 3 RWs would draw ~40% of available power during maneuvers
- MTQ-only achieves only ~5-10° pointing (insufficient for imaging)

**3+1 Solution**:
- Single RW + 3 compact MTQs fits within volume budget
- Power draw reduced by ~2/3 for attitude actuators
- Achieves <1° pointing with planner

**Claim**: "3+1 enables sub-degree imaging missions at the 1U scale that were previously constrained by volume and power limitations."

---

#### VALIDATED EXAMPLE B: Constellation Economics

**Scenario**: 200-satellite LEO IoT constellation, each needing 2° pointing accuracy.

**Cost Analysis**:
| Item | 3×RW Config | 3+1 Config | Savings |
|------|-------------|------------|---------|
| RW hardware | 3 × $7,500 × 200 = **$4.5M** | 1 × $7,500 × 200 = **$1.5M** | **$3.0M** |
| Extra mass | 2 RW × 100g × 200 = 40 kg | — | 40 kg |
| Launch cost | 40 kg × $5,500/kg = **$220k** | — | **$220k** |
| **TOTAL** | | | **$3.2M+** |

**Additional factors**:
- 50% of ADCS failures are RW-related → fewer RWs = higher reliability
- Reduced power draw extends battery life or allows smaller solar panels
- Reduced volume allows larger payload or additional subsystems

**Claim**: "For a 200-satellite constellation, 3+1 architecture saves over $3 million in hardware and launch costs while potentially improving reliability by reducing moving-part failure modes."

---

#### VALIDATED EXAMPLE C: Power-Constrained Eclipse Mission

**Scenario**: 3U aurora-observation CubeSat operating through long eclipse periods at high latitude.

**Power Analysis**:
- Typical CubeSat RW power: 0.4-0.7W each (from CubeSpace datasheets)
- 3 RWs: ~1.5W continuous
- 1 RW + MTQs: ~0.5W + ~0.6W = ~1.1W (MTQs only active intermittently)
- **Net savings**: ~0.4-1.0W continuous

**Eclipse Impact**:
- 40-minute eclipse at polar orbit
- 1W savings × 0.67 hours = **0.67 Wh per eclipse**
- Typical 3U battery: 20-40 Wh
- Savings = **2-3% of battery per eclipse** → significant for multi-eclipse operations

**Claim**: "3+1 reduces ADCS power consumption by 25-40%, enabling extended eclipse operations or smaller battery sizing for polar and high-inclination missions."

---

#### EXAMPLE D: BeaverCube-2 (Real Mission)

**STATUS**: Patrick to provide specific details:
- What was the binding constraint? (mass/power/volume/cost)
- What pointing accuracy was required?
- What would have been sacrificed for 3RW?

---

## Session 3: Paper Sharpening & Experiment Planning
**Date**: 2026-01-23 (continued)

### Key Decisions Made

#### 1. GENERALIZABILITY IS THE CORE THESIS
**Critical insight**: Papers must show VERSATILITY across diverse configurations, not just BC2.

- **3+1 Paper**: BC2 is primary example (real mission)
- **Package Paper**: Must show MULTIPLE spacecraft (1U, 3U, 6U, different actuators)
- **Planner Paper**: Must work across MTQ-only, 3+1, 3+3, different orbits/goals
- **Generalized Control**: Allocation works across ALL configs (that's the point!)

The message is: "Configuration replaces custom code" — one framework handles everything.

#### 2. LP vs QP Resolution
From research/ folder exploration, extensive constraint research completed:

**Best QP Constraints (120s closed-loop test)**:
| Method | θ_final | Notes |
|--------|---------|-------|
| QP unconstrained | 2.35° | Best but no physics guarantee |
| 3b-Sign critical | 2.35° | Tied best, per-axis sign check |
| Phase-aware | 2.69° | Adapts to convergence phase |
| LP | 2.94° | Reliable baseline |
| 1a-Power brake only | 3.11° | Physics-based, nearly as good |

**Key insight**: Constraints must be CONDITIONAL on controller intent:
- When braking (P_des < 0): Constrain energy
- When accelerating: Let controller work

**Failed approaches** (caused 70-125° errors): Pure Lyapunov, always-on power bound

**STATUS**: Results are preliminary - need full MC validation before publishing.

#### 3. Test Infrastructure Discovered
`testing/paper_todo_tests/` has **90 tests** covering 89% of paper TODOs!

Key test files:
- `test_todo_data_lp_qp_comparison.py` - 14 tests
- `test_todo_sim_monte_carlo.py` - MC infrastructure
- `test_todo_data_desaturation.py` - 18 tests
- `test_todo_sim_scenarios.py` - pointing, tracking, failure

Run with: `pytest testing/paper_todo_tests/ -v -s`

#### 4. BC2 Parameters (from codebase)
```python
# ADCS/satellite_factory/satellites/create_cubesats.py
mass = 4  # kg
J = [[0.0314, 0.0001, -0.0067],
     [0.0001, 0.0341, -0.0001],
     [-0.0067, -0.0001, 0.0100]]  # kg·m²
# 3U, ISIS MTQs, CubeWheel SmallPlus RW (z-axis)
# Boresight: y-axis (camera)
# Mission: Ground target tracking (REDUCED-ATTITUDE goals)
```

Factory functions available:
- `create_beavercube2_cubesat()` - 3+1
- `create_3_3_beavercube2_cubesat()` - 3+3
- `create_beavercube1_cubesat()` - 3+0

#### 5. Planner Benchmarking
ALTRO is the final choice, but will benchmark against alternatives in `papers/planner_comparison/`:
- Convex MPC
- Direct collocation
- Pseudospectral
- Polynomial shaping
- Eigenaxis trapezoidal

### Thesis Results Incorporated (from planning.tex, disturbance.tex)

**Monte Carlo Results**:
- Single 180° slew: MTQ-only 73% within 10°; 3MTQ+1RW 96% within 1°
- Goal-set slews: MTQ-only 67% within 1°; 3MTQ+1RW 96% within 1°
- Multi-target: 3MTQ+1RW 98%+ within 10°, mean 0.45°, median 0.03°

**Disturbance Context (thesis Figure 6.X)**:
- No compensation: ~180° error (uncontrolled)
- All-in-one disturbance: 5-20° error
- Full disturbance model: 0-4° error

### Detailed Experiment Lists Added to Papers

Each paper now has comprehensive experiment lists in the header comments:
- **3+1 Paper**: Sets A-D (core comparison, momentum mgmt, mission scenarios, sensitivity)
- **Planner Paper**: Sets A-F (baselines, goal formulation, multi-target, graceful degradation, computational, sensitivity)
- **Package Paper**: Sets A-E (usability, validation, benchmarks, Basilisk comparison, demos)
- **Generalized Control**: Sets A-E (LP/QP, controllability, modular demo, desaturation, MC validation)

### Open Items for Patrick

| Item | Status | Action Needed |
|------|--------|---------------|
| BC2 pointing requirement | Unknown | Get from mission team |
| BC2 launch timeline | Unknown | Get from mission team |
| QP constraint MC validation | Undecided | Decide: full MC or representative? |
| Basilisk comparison | Not started | Need to install Basilisk |
| Pi 4 timing benchmarks | Not started | Need hardware access |

### Papers Updated This Session
1. All four main2.tex files updated with:
   - Detailed experiment lists
   - Thesis results references
   - BC2 parameters (where appropriate)
   - Emphasis on GENERALIZABILITY (diverse configs)
   - QP constraint research findings (with caveats)

---

