"""Plot attitude and rate knowledge error in individual and magnitude modes."""

# Select the backend for this example: "rust", "cpp", or "python".
backed = "cpp"

if backed not in ("rust", "cpp", "python"):
    raise ValueError("backed must be rust, cpp, or python")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import AbsoluteKnowledgeErrorQuaternionPlot, plot
from GenADCS.backends.shared.plotting import AbsoluteKnowledgeErrorQuaternionPlot, plot


def main():
    result = run_example(backed=backed)
    plot(result, AbsoluteKnowledgeErrorQuaternionPlot(option="individual"), AbsoluteKnowledgeErrorQuaternionPlot(option="magnitude"), layout=(2, 1),
         title="Attitude knowledge error", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
