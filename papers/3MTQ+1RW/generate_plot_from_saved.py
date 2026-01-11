import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc, plot_convergence_histogram_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

if __name__ == "__main__":
    # results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_LP_mc_100_20260110_163520")
    # results_lp = results_lp[0]
    # results_qpw = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPW_mc_100_20260110_164621")
    # results_qpw = results_qpw[0]

    results_lp = load_data("papers/3MTQ+1RW/output_data/3MTQ+2RW_LP_mc_100_20260110_163915")
    results_lp = results_lp[0]
    results_qpw = load_data("papers/3MTQ+1RW/output_data/3MTQ+2RW_QPW_mc_100_20260110_195156")
    results_qpw = results_qpw[0]

    plot_target_tracking_mc(full_results=results_lp, title="3 MTQ + 2 RW LP MC:100 $Mu = \\alpha\\tau$")
    plot_convergence_histogram_mc(full_results=results_lp, title="3 MTQ + 2 RW LP MC:100 $Mu = \\alpha\\tau$")
    plot_target_tracking_mc(full_results=results_qpw, title="3 MTQ + 2 RW QPW MC:100 $|W(Mu - \\tau)|^2$")
    plot_convergence_histogram_mc(full_results=results_qpw, title="3 MTQ + 2 RW QPW MC:100 $|W(Mu - \\tau)|^2$")
    create_close_all_button_window()