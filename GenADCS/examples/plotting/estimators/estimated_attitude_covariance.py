"""Simulate with AttitudeMEKF and plot estimated attitude covariance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import EstimatedAttitudeCovariancePlot, plot


def main():
    result = run_example()
    plot(result,
         EstimatedAttitudeCovariancePlot(option="individual"),
         EstimatedAttitudeCovariancePlot(option="trace"),
         EstimatedAttitudeCovariancePlot(option="eigenvalues"),
         layout=(1, 3), title="Estimated attitude covariance", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
