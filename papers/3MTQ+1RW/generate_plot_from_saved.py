import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting_mc.plot_controller_compare_mc import plot_target_tracking_mc_compare, plot_convergence_histogram_mc_compare
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

if __name__ == "__main__":
    # results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_20260110_163520")
    # results_lp = results_lp[0]
    # results_qpw = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPW_mc_100_20260110_164621")
    # results_qpw = results_qpw[0]

    results_lovera = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_Lovera_mc_100_1000s_20260118_022947")[0]
    results_wisniewski = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_Wisniewski_mc_100_1000s_20260118_022551")[0]
    results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_1000s_20260118_023305")[0]
    results_hogan_3 = load_data("papers/3MTQ+1RW/output_data/3MTQ+3RW_MTQ_w_RW_mc_100_1000s_20260118_025715")[0]
    results_lp_3 = load_data("papers/3MTQ+1RW/output_data/3MTQ+3RW_MTQ_w_RW_LP_mc_100_1000s_20260118_162607")[0]

    plot_target_tracking_mc_compare(results_lovera, results_lp, boresight = np.array([0, 1, 0]), title="3 MTQ + 0 RW Lovera vs 3 MTQ + 1 RW LP", label_A="Lovera", label_B="LP")
    plot_convergence_histogram_mc_compare(results_lovera, results_lp, boresight = np.array([0, 1, 0]), title="Convergence Histogram: 3 MTQ + 0 RW Lovera vs 3 MTQ + 1 RW LP | 1000s", label_A="Lovera", label_B="LP")
    plot_target_tracking_mc_compare(results_wisniewski, results_lp, boresight = np.array([0, 1, 0]), title="3 MTQ + 0 RW Wisniewski vs 3 MTQ + 1 RW LP", label_A="Wisniewski", label_B="LP")
    plot_convergence_histogram_mc_compare(results_wisniewski, results_lp, boresight = np.array([0, 1, 0]), title="Convergence Histogram: 3 MTQ + 0 RW Wisniewski vs 3 MTQ + 1 RW LP | 1000s", label_A="Wisniewski", label_B="LP")
    plot_target_tracking_mc_compare(results_hogan_3, results_lp, boresight = np.array([0, 1, 0]), title="3 MTQ + 3 RW vs 3 MTQ + 1 RW LP", label_A="3 MTQ + 3 RW", label_B="LP")
    plot_convergence_histogram_mc_compare(results_hogan_3, results_lp, boresight = np.array([0, 1, 0]), title="Convergence Histogram: 3 MTQ + 3 RW vs 3 MTQ + 1 RW LP | 1000s", label_A="3 MTQ + 3 RW", label_B="LP")
    plot_target_tracking_mc_compare(results_hogan_3, results_lp_3, boresight = np.array([0, 1, 0]), title="3 MTQ + 3 RW vs 3 MTQ + 3 RW LP", label_A="3 MTQ + 3 RW", label_B="LP")
    plot_convergence_histogram_mc_compare(results_hogan_3, results_lp_3, boresight = np.array([0, 1, 0]), title="Convergence Histogram: 3 MTQ + 3 RW vs 3 MTQ + 3 RW LP | 1000s", label_A="3 MTQ + 3 RW", label_B="LP")
    create_close_all_button_window()