"""Compare the four attitude filters on one seeded sensor and orbit scenario."""

import matplotlib.pyplot as plt

from _comparison_common import ADCS, run


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, filter_type in (("EKF", ADCS.EKF), ("MEKF", ADCS.MEKF),
                              ("UKF", ADCS.UKF), ("SRUKF", ADCS.SRUKF)):
        time, error = run(filter_type)
        ax.plot(time, error, label=name)
    ax.set(xlabel="Time (s)", ylabel="Attitude error (deg)",
           title="EKF, MEKF, UKF and SRUKF on the same scenario")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
