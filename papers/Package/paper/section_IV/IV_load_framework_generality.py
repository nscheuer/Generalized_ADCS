"""Load and plot the saved Section IV framework-generality campaign."""

from __future__ import annotations

try:
    from .IV_framework_generality import (
        OUTPUT_DIR,
        OUTPUT_NAMES,
        SATELLITES,
        plot_comparisons,
    )
except ImportError:  # Support running this file directly from section_IV.
    from IV_framework_generality import OUTPUT_DIR, OUTPUT_NAMES, SATELLITES, plot_comparisons

from ADCS.helpers.save_and_load.save_and_load import load_data


def main():
    """Load the newest saved result for every spacecraft and replot it."""
    all_results = {}
    for name in SATELLITES:
        candidates = sorted(OUTPUT_DIR.glob(f"{OUTPUT_NAMES[name]}_*/"))
        if not candidates:
            raise FileNotFoundError(
                f"No saved results for {name} found in {OUTPUT_DIR}. "
                "Run IV_framework_generality.py first."
            )
        path = candidates[-1]
        all_results[name] = load_data(path)[0]
        print(f"Loaded {name} results from {path}")
    plot_comparisons(all_results)
    return all_results


if __name__ == "__main__":
    main()
