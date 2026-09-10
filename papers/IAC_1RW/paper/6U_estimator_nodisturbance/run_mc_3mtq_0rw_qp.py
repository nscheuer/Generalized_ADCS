"""Run one diagnostic 6U 3-MTQ + 0-RW QP simulation (no Monte Carlo)."""

from _6u_mc_common import run_diagnostic


if __name__ == "__main__":
    run_diagnostic(number_rw=0, allocator="qp", label="3 MTQ + 0 RW (QP)")
