import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting_mc.plot_controller_compare_mc import plot_target_tracking_mc_compare, plot_convergence_histogram_mc_compare
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

if __name__ == "__main__":
    results_planner = load_data("papers/Planner/output_data/3MTQ+1RW_ALTRO_100_1000s_reduced_20260128_165403")[0]

    plot_target_tracking_mc(results_planner, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner Target Tracking MC")
    plot_convergence_histogram_mc(results_planner, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner Convergence Histogram MC")

    create_close_all_button_window()