"""Generate Table 6 cell: 3MTQ+1RW, vector goal, LP allocator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from papers.Generalized_ACS.paper.section_IV_helpers import run_table6_cell


if __name__ == "__main__":
    run_table6_cell("3MTQ+1RW", "vector", "lp", "table6_3mtq_1rw_vector_lp")
