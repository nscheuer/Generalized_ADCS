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


    results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_4000s_20260121_195055")[0]
    result_lp_old = load_data("papers/3MTQ+1RW/output_data_old/3MTQ+1RW_LP_mc_100_1000s_20260118_201308")[0]

    plot_target_tracking_mc(full_results=results_lp, body_boresight=np.array([0, 1, 0]), title="3 MTQ + 1 RW LP MC:100")
    plot_convergence_histogram_mc(full_results=results_lp, body_boresight=np.array([0, 1, 0]), title="3 MTQ + 1 RW LP")
    plot_target_tracking_mc(full_results=result_lp_old, body_boresight=np.array([0, 0, 1]), title="3 MTQ + 1 RW LP MC:100 Old")
    plot_convergence_histogram_mc(full_results=result_lp_old, body_boresight=np.array([0, 0, 1]), title="3 MTQ + 1 RW LP Old")
    create_close_all_button_window()