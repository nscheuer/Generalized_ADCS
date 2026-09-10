"""Load and plot a saved rapid actuator-architecture trade study.

Run from the repository root with::

    python papers/Package/paper/section_IV/IV_load_rapid_architecture_trade.py
"""

from ADCS import SimulationResults

try:
    from .IV_rapid_architecture_trade import (
        ARCHITECTURES,
        OUTPUT_DIR,
        OUTPUT_NAMES,
        plot_comparisons,
    )
except ImportError:  # Support running this file directly from section_IV.
    from IV_rapid_architecture_trade import (
        ARCHITECTURES,
        OUTPUT_DIR,
        OUTPUT_NAMES,
        plot_comparisons,
    )


def main():
    """Load the newest saved campaign and save/display its comparison plots."""
    all_results = {}
    for architecture in ARCHITECTURES:
        candidates = sorted(OUTPUT_DIR.glob(f"{OUTPUT_NAMES[architecture]}_*.sim"))
        if not candidates:
            raise FileNotFoundError(
                f"No saved results for {architecture} found in {OUTPUT_DIR}. "
                "Run IV_rapid_architecture_trade.py first."
            )
        path = candidates[-1]
        all_results[architecture] = SimulationResults.load(path)
        print(f"Loaded {architecture} results from {path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
