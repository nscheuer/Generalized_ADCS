# Paste-ready: Zhu et al. direct engagement + corrected references

## BibTeX (replaces any "Li et al." attribution for 2023-0935)

```bibtex
@inproceedings{Zhu2023,
  author    = {Zhu, Yufei and Sutherland, Richard and Girard, Anouck and Gilchrist, Brian},
  title     = {Attitude Control of a {3U} {CubeSat} with Combination of Magnetorquers
               and Reaction Wheels},
  booktitle = {AIAA SciTech Forum},
  year      = {2023},
  number    = {AIAA 2023-0935},
  doi       = {10.2514/6.2023-0935}
}

@inproceedings{Sahnow2006,
  author    = {Sahnow, David J. and Kruk, Jeffrey W. and Ake, Thomas B. and Andersson,
               B-G and Berman, Alice and Blair, William P. and Boyer, Robert and
               Caplinger, James and Calvani, Humberto and Civeit, Thomas and
               Van Dyke Dixon, W. and England, Martin N. and Kaiser, Mary Elizabeth and
               Kochte, Mark and Moos, H. Warren and Roberts, Bryce A.},
  title     = {Operations with the new {FUSE} observatory: three-axis control with one
               reaction wheel},
  booktitle = {Proc. SPIE 6266, Observatory Operations: Strategies, Processes, and Systems},
  year      = {2006},
  doi       = {10.1117/12.673153}
}
```
(Verify the SPIE DOI digits at proof stage; volume/title/authors are from the paper's own
title page.)

## Prior-art paragraph (Section II, direct engagement)

The 3\,MTQ\,+\,1\,RW complement itself is not new: Zhu et al.\ \cite{Zhu2023} propose
precisely this configuration for a 3U CubeSat, as the trade between a magnetorquer-only LQR
and a three-wheel PD, and demonstrate nominal pointing in simulation. What that study --- and,
to our knowledge, the literature following it --- does not address is where the configuration
\emph{fails}: single-trajectory and small-sample demonstrations cannot expose a failure set
that occupies $\sim$11\% of the goal--geometry space and is invisible from any individual
converged run. The present campaign is complementary in exactly that sense. We confirm the
nominal-cell result at scale ($n{=}100$: 82\% within $1^\circ$, median $0.25^\circ$), and then
characterize what Zhu et al.\ could not see: a geometrically clustered divergent set, its
mechanism (transverse-authority exhaustion in dump-starved field geometry, not wheel
saturation --- Sec.~VI-C), a pre-flight screen for it (Sec.~VI-E), and the momentum-bias
ceiling that bounds the operating point (Sec.~VI-D). Flight precedent for one-wheel-plus-
magnetics operation exists at much larger scale: FUSE operated for years with a single
reaction wheel and torquer bars at arcsecond-class jitter \cite{Sahnow2006}, albeit with
severely restricted attitude availability --- a restriction whose small-satellite analogue is
precisely the availability structure this paper maps.

## One-sentence version (if space forces it)

Zhu et al.\ \cite{Zhu2023} propose this exact complement and demonstrate its nominal
performance; we confirm that result at scale and characterize its failure set, mechanism,
and pre-flight screen --- the questions a demonstration cannot reach.
