"""Run one diagnostic 6U 3-MTQ + 1-RW LP simulation (no Monte Carlo)."""

from _6u_mc_common import run_diagnostic


if __name__ == "__main__":
    run_diagnostic(number_rw=1, allocator="lp", label="3 MTQ + 1 RW (LP)")
