"""Simulate AttitudeMEKF and compare measurement innovation covariance views."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import MeasurementInnovationCovariancePlot, plot


def main():
    result = run_example()
    plot(result,
         MeasurementInnovationCovariancePlot(option="individual"),
         MeasurementInnovationCovariancePlot(option="trace"),
         MeasurementInnovationCovariancePlot(option="eigenvalues"),
         layout=(1, 3), title="Measurement Innovation Covariance", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result


if __name__ == "__main__":
    main()
