# Verified bibliography entries (2026-08-19)

## FUSE one-wheel operations — VERIFIED from the paper's own title page

> D. J. Sahnow, J. W. Kruk, T. B. Ake, B-G Andersson, A. Berman, W. P. Blair, R. Boyer,
> J. Caplinger, H. Calvani, T. Civeit, W. Van Dyke Dixon, M. N. England, M. E. Kaiser,
> M. Kochte, H. W. Moos, and B. A. Roberts,
> "Operations with the new FUSE observatory: three-axis control with one reaction wheel,"
> Proc. SPIE 6266 (Observatory Operations: Strategies, Processes, and Systems), 2006.

Affiliations: JHU Physics & Astronomy; Computer Sciences Corp.; CNES; UC Berkeley SSL.
Source PDF: https://archive.stsci.edu/fuse/papers/spie6266/6266-2_submitted2.pdf
(title page read directly; author order transcribed verbatim, 16 authors)

USABILITY NOTE for the prior-art paragraph: the abstract states FUSE operated "with only one
reaction wheel" with jitter requirements of +-1 arcsec pitch / +-10 arcsec yaw / +-1 deg roll
-- so it is a genuine one-wheel-plus-magnetics operations precedent, NOT merely a gyro story.
The earlier concern (HST = gyro story) does not apply here; FUSE is the citation to carry.

## AIAA 2023-0935 ("the Li paper" -- author list is actually Zhu et al.)

> Y. Zhu, R. Sutherland, A. Girard, and B. Gilchrist,
> "Attitude Control of a 3U CubeSat with Combination of Magnetorquers and Reaction Wheels,"
> AIAA SciTech Forum, 2023. doi:10.2514/6.2023-0935.

University of Michigan. Published online 2023-01-19.
NOTE: if the bibliography stub says "Li et al." for 2023-0935, that attribution is WRONG --
first author is Zhu. (If a separate "Li et al." entry exists it refers to a different paper
and still needs its own verification.)
Sources: https://arc.aiaa.org/doi/10.2514/6.2023-0935 (403 to fetchers; metadata via
ResearchGate mirror + search), https://www.researchgate.net/publication/367319909

RELEVANCE: proposes exactly this paper's third configuration -- PD on 3 MTQ + one pitch-axis
RW as the trade between MTQ-only LQR and 3-wheel PD. This is the closest prior art to the
nominal-cell claim and should be engaged directly, not just cited.

## Li et al. 2013 — VERIFIED (separate paper from Zhu; second direct prior-art hit)

> J. Li, M. Post, T. Wright, and R. Lee,
> "Design of Attitude Control Systems for CubeSat-Class Nanosatellite,"
> Journal of Control Science and Engineering, vol. 2013, art. 657182, 2013.
> doi:10.1155/2013/657182

York University, Dept. of Earth & Space Science and Engineering. Full names: Junquan Li,
Mark Post, Thomas Wright, Regina Lee.

RELEVANCE UPGRADE: their fine-pointing mode is "three magnetorquers and one reaction wheel
along the pitch axis" -- the exact complement, on a 1U, a DECADE before Zhu 2023. The
prior-art paragraph strengthens accordingly: the configuration has been proposed at least
twice across ten years (Li 2013 1U, Zhu 2023 3U), both as nominal demonstrations; what the
literature has not asked is where it fails, which is precisely this paper's contribution.
Sources: Wiley 10.1155/2013/657182 (paywalled), Semantic Scholar, York nanosatellite lab
publications page.

## Dual-spin MPC (resolved by Patrick): Halverson & Caverly, arXiv:2506.07858, preprint to JGCD.

## Wheel-failure statistic: CUT (Patrick, 2026-08-19)

FUSE carries the failure narrative concretely (four wheels, successive failures, years of
one-wheel operations); an aggregate reliability figure adds nothing a B4 audience needs and
would itself require a defensible source. No further chase.
