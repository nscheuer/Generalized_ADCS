"""Load and display saved 3-MTQ + 0-RW QP diagnostics."""

from _6u_mc_common import load_diagnostic


if __name__ == "__main__":
    load_diagnostic(number_rw=0, allocator="qp", label="3 MTQ + 0 RW (QP)")
