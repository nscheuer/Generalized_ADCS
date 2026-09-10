"""Load and display saved 3-MTQ + 3-RW LP diagnostics."""

from _6u_mc_common import load_diagnostic


if __name__ == "__main__":
    load_diagnostic(number_rw=3, allocator="lp", label="3 MTQ + 3 RW (LP)")
