"""Simulate with AttitudeMEKF and plot estimated rate covariance."""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "python"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import EstimatedRateCovariancePlot, plot


def main():
    result = run_example(backed=backed)
    plot(result,
         EstimatedRateCovariancePlot(option="individual"),
         EstimatedRateCovariancePlot(option="trace"),
         EstimatedRateCovariancePlot(option="eigenvalues"),
         layout=(1, 3), title="Estimated rate covariance", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
