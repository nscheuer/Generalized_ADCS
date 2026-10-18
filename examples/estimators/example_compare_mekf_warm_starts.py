"""Show how single-frame warm starts help an MEKF with a large initial error."""

import matplotlib.pyplot as plt

from _comparison_common import ADCS, run


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, start in (
        ("No warm start", None),
        ("TRIAD", ADCS.WarmStart(method="triad")),
        ("QUEST", ADCS.WarmStart(method="quest")),
        ("q-method", ADCS.WarmStart(method="q_method")),
    ):
        # This hardcoded quaternion is 170 degrees from truth. The warm starts
        # instead solve their initial attitude from the first sensor readings.
        time, error = run(ADCS.MEKF, warm_start=start, initial_half_angle_deg=170.0)
        ax.plot(time, error, label=label)
    ax.set(xlabel="Time (s)", ylabel="Attitude error (deg)",
           title="MEKF with and without single-frame warm starts on the same scenario")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
