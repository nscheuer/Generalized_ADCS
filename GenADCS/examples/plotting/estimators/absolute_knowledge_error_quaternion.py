"""Plot attitude and rate knowledge error in individual and magnitude modes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from GenADCS.examples.plotting.estimators._setup import run_example
from GenADCS.backends.shared.plotting import AbsoluteKnowledgeErrorQuaternionPlot, plot
from GenADCS.backends.shared.plotting import AbsoluteKnowledgeErrorQuaternionPlot, plot


def main():
    result = run_example()
    plot(result, AbsoluteKnowledgeErrorQuaternionPlot(option="individual"), AbsoluteKnowledgeErrorQuaternionPlot(option="magnitude"), layout=(2, 1),
         title="Attitude knowledge error", show=True)
    import matplotlib.pyplot as plt
    plt.show()
    return result

if __name__ == "__main__":
    main()
