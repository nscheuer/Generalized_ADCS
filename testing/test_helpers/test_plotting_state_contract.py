import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from ADCS.helpers.plotting.plot_controller import plot_control
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.state import State


def test_plot_helpers_accept_state_lists_with_agg_backend():
    time = np.array([0.0, 1.0])
    states = [
        State(w=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.1]),
        State(w=[0.1, -0.1, 0.2], q=[1.0, 0.0, 0.0, 0.0], h=[0.2]),
    ]
    controls = np.zeros((2, 3))

    try:
        plot_state_comparison(time=time, state_hist=states)
        plot_control(time=time, u_hist=controls)
        assert len(plt.get_fignums()) == 2
    finally:
        plt.close("all")
