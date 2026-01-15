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

    results_g0 = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPG0.1_mc_100_20260110_202754")[0]
    results_g10 = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPG10_mc_32_20260111_012751")[0]
    results_g100 = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_QPG100_mc_32_20260111_012909")[0]

    plot_target_tracking_mc(full_results=results_g0, title="3 MTQ + 1 RW QPG $\\gamma = 0.1$ MC:100 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    plot_convergence_histogram_mc(full_results=results_g0, title="3 MTQ + 1 RW QPG $\\gamma = 0.1$ MC:100 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    plot_target_tracking_mc(full_results=results_g10, title="3 MTQ + 1 RW QPG $\\gamma = 10$ MC:32 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    plot_convergence_histogram_mc(full_results=results_g10, title="3 MTQ + 1 RW QPG $\\gamma = 10$ MC:32 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    plot_target_tracking_mc(full_results=results_g100, title="3 MTQ + 1 RW QPG $\\gamma = 100$ MC:32 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    plot_convergence_histogram_mc(full_results=results_g100, title="3 MTQ + 1 RW QPG $\\gamma = 100$ MC:32 $|(I + \\gamma \\frac{\\omega \\omega^{\\top}}{|\\omega|^2})(Mu - \\tau)|^2$")
    create_close_all_button_window()