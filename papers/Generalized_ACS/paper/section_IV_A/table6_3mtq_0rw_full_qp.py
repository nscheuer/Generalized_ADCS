"""Generate Table 6 cell: 3MTQ+0RW, full goal, QP allocator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from papers.Generalized_ACS.paper.section_IV_helpers import run_table6_cell


if __name__ == "__main__":
    run_table6_cell("3MTQ+0RW", "full", "qp", "table6_3mtq_0rw_full_qp")
