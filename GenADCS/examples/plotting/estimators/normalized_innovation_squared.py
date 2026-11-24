"""Plot normalized innovation squared for the shared MEKF estimator simulation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import NormalizedInnovationSquaredPlot, plot


def main():
    result = run_example()
    plot(result, NormalizedInnovationSquaredPlot(), show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result


if __name__ == "__main__":
    main()
