import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting_mc.plot_controller_compare_mc import plot_target_tracking_mc_compare, plot_convergence_histogram_mc_compare
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

if __name__ == "__main__":
    results_wis = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_Wisniewski_mc_100_1000s_20260121_200028")[0]
    results_lovera = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_Lovera_mc_100_1000s_20260121_200244")[0]
    results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_1000s_20260121_195055")[0]
    results_planner = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_ALTRO_mc_100_1000s_20260122_204611")[0]
    results_3 = load_data("papers/3MTQ+1RW/output_data/3MTQ+3RW_MTQ_w_RW_LP_mc_100_1000s_20260121_195752")[0]

    # plot_target_tracking_mc(results_wis, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW Wisniewski Controller Target Tracking MC Results")
    # plot_convergence_histogram_mc(results_wis, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW Wisniewski Controller Convergence Histogram MC Results")

    # plot_target_tracking_mc(results_lovera, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW Lovera Controller Target Tracking MC Results")
    # plot_convergence_histogram_mc(results_lovera, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW Lovera Controller Convergence Histogram MC Results")

    # plot_target_tracking_mc(results_lp, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP Controller Target Tracking MC Results")
    # plot_convergence_histogram_mc(results_lp, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP Controller Convergence Histogram MC Results")

    # plot_target_tracking_mc(results_planner, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner Target Tracking MC Results")
    # plot_convergence_histogram_mc(results_planner, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner Convergence Histogram MC Results")

    # plot_target_tracking_mc(results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+3RW+LP Controller Target Tracking MC Results")
    # plot_convergence_histogram_mc(results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+3RW+LP Controller Convergence Histogram MC Results")

    plot_target_tracking_mc_compare(results_lp, results_wis, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs Wisniewski Controller Target Tracking MC Comparison")
    plot_convergence_histogram_mc_compare(results_lp, results_wis, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs Wisniewski Controller Convergence Histogram MC Comparison")

    plot_target_tracking_mc_compare(results_lp, results_lovera, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs Lovera Controller Target Tracking MC Comparison")
    plot_convergence_histogram_mc_compare(results_lp, results_lovera, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs Lovera Controller Convergence Histogram MC Comparison")

    plot_target_tracking_mc_compare(results_lp, results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs 3MTQ+3RW+LP Controller Target Tracking MC Comparison")
    plot_convergence_histogram_mc_compare(results_lp, results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW LP vs 3MTQ+3RW+LP Controller Convergence Histogram MC Comparison")

    plot_target_tracking_mc_compare(results_planner, results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner vs 3MTQ+3RW+LP Controller Target Tracking MC Comparison")
    plot_convergence_histogram_mc_compare(results_planner, results_3, body_boresight=np.array([0, 1, 0]), title="3MTQ+1RW ALTRO Planner vs 3MTQ+3RW+LP Controller Convergence Histogram MC Comparison")
    create_close_all_button_window()