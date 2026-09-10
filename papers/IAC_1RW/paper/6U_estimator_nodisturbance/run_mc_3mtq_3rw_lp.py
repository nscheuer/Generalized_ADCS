"""Run one diagnostic 6U 3-MTQ + 3-RW LP simulation (no Monte Carlo)."""

from _6u_mc_common import run_diagnostic


if __name__ == "__main__":
    run_diagnostic(number_rw=3, allocator="lp", label="3 MTQ + 3 RW (LP)")
