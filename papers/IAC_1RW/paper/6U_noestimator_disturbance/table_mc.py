"""Print the convergence summary table from the latest 6U MC results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
import table_mc as _table_mc
_table_mc.CAMPAIGN_PREFIX = "6u_noestimator_disturbance"
_table_mc.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

if __name__ == "__main__":
    _table_mc.main()
