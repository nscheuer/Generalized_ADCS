"""Generate benchmark plots and PR-comment markdown."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GREEN = "#2da44e"
YELLOW = "#bf8700"
RED = "#cf222e"


@dataclass(frozen=True)
class BenchmarkRatio:
    category: str
    name: str
    ratio: float

    @property
    def severity(self) -> str:
        if 0.8 <= self.ratio <= 1.2:
            return "green"
        if 0.5 <= self.ratio < 0.8 or 1.2 < self.ratio <= 1.5:
            return "yellow"
        return "red"


def ratio_color(ratio: float) -> str:
    if 0.8 <= ratio <= 1.2:
        return GREEN
    if 0.5 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
        return YELLOW
    return RED


def ratio_badge(ratio: float) -> str:
    color = {"green": "brightgreen", "yellow": "yellow", "red": "red"}[BenchmarkRatio("", "", ratio).severity]
    return f"![{ratio:.2f}x](https://img.shields.io/badge/{ratio:.2f}x-{color})"


def load_ratios(category: str, current_path: Path, baseline_path: Path) -> list[BenchmarkRatio]:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    ratios = []
    for name, current_entry in current["benchmarks"].items():
        if name not in baseline["benchmarks"]:
            continue
        baseline_normalized = baseline["benchmarks"][name]["normalized_seconds"]
        ratio = current_entry["normalized_seconds"] / baseline_normalized
        ratios.append(BenchmarkRatio(category=category, name=name, ratio=ratio))
    return ratios


def category_stats(ratios: list[BenchmarkRatio]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    by_category: dict[str, list[float]] = {}
    for ratio in ratios:
        by_category.setdefault(ratio.category, []).append(ratio.ratio)

    for category, values in by_category.items():
        stats[category] = {
            "lowest": min(values),
            "average": sum(values) / len(values),
            "highest": max(values),
        }
    return stats


def write_plot(stats: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = list(stats)
    labels = []
    values = []
    colors = []
    for category in categories:
        for metric in ("lowest", "average", "highest"):
            value = stats[category][metric]
            labels.append(f"{category} {metric}")
            values.append(value)
            colors.append(ratio_color(value))

    height = max(2.0, 0.34 * len(labels) + 0.9)
    fig, ax = plt.subplots(figsize=(6.4, height))
    y_positions = range(len(labels))
    bars = ax.barh(y_positions, values, color=colors)

    ax.axvspan(0.8, 1.2, color=GREEN, alpha=0.08, linewidth=0)
    ax.axvspan(0.5, 0.8, color=YELLOW, alpha=0.08, linewidth=0)
    ax.axvspan(1.2, 1.5, color=YELLOW, alpha=0.08, linewidth=0)
    ax.axvline(1.0, color="#57606a", linestyle="--", linewidth=1.0)
    ax.axvline(1.5, color=RED, linestyle=":", linewidth=1.0)

    ax.set_yticks(list(y_positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("calibrated multiplier vs baseline")
    ax.set_title("Benchmark Multiplication Factors")
    ax.set_xlim(0, max(1.6, max(values) * 1.15))
    ax.grid(axis="x", alpha=0.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}x",
            va="center",
            fontsize=8,
            color="#24292f",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=90, pil_kwargs={"optimize": True})
    plt.close(fig)


def outside_green(ratios: list[BenchmarkRatio]) -> list[BenchmarkRatio]:
    flagged = [ratio for ratio in ratios if ratio.severity != "green"]
    return sorted(
        flagged,
        key=lambda item: (
            item.ratio < 1.0,
            -item.ratio if item.ratio >= 1.0 else item.ratio,
        ),
    )


def write_markdown(ratios: list[BenchmarkRatio], plot_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_plot = base64.b64encode(plot_path.read_bytes()).decode("ascii")
    flagged = outside_green(ratios)

    lines = [
        "<!-- generalized-adcs-benchmark-report -->",
        "## Benchmark Report",
        "",
        f'<img alt="Benchmark multiplication factors" src="data:image/png;base64,{encoded_plot}">',
        "",
        "### Tests Outside Green Range",
        "",
    ]

    if flagged:
        lines.extend(["| status | category | test | multiplier |", "| --- | --- | --- | --- |"])
        for item in flagged:
            lines.append(f"| {ratio_badge(item.ratio)} | `{item.category}` | `{item.name}` | `{item.ratio:.2f}x` |")
    else:
        lines.append("All benchmark tests are within the green `0.8x` to `1.2x` range.")

    lines.extend(
        [
            "",
            "Color bands: green `0.8x-1.2x`, yellow `0.5x-0.8x` or `1.2x-1.5x`, red outside that range.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(ratios: list[BenchmarkRatio], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "categories": category_stats(ratios),
        "tests": [
            {
                "category": ratio.category,
                "name": ratio.name,
                "ratio": ratio.ratio,
                "severity": ratio.severity,
            }
            for ratio in ratios
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark report artifacts.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratios = load_ratios(args.category, args.current, args.baseline)
    if not ratios:
        raise SystemExit("No benchmark ratios found")

    stats = category_stats(ratios)
    write_plot(stats, args.plot)
    write_markdown(ratios, args.plot, args.markdown)
    write_summary_json(ratios, args.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
