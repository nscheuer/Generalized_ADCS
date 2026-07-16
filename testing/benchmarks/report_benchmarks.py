"""Generate benchmark plots and PR-comment markdown."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GREEN = "#2da44e"
YELLOW = "#bf8700"
RED = "#cf222e"
OUTLINE = "#24292f"


def rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        alpha,
    )


def css_rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    return (
        f"rgba({int(value[0:2], 16)}, "
        f"{int(value[2:4], 16)}, "
        f"{int(value[4:6], 16)}, "
        f"{alpha})"
    )


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
    highest_values = [stats[category]["highest"] for category in categories]
    average_values = [stats[category]["average"] for category in categories]
    lowest_values = [stats[category]["lowest"] for category in categories]

    height = max(2.2, 0.55 * len(categories) + 1.7)
    fig, ax = plt.subplots(figsize=(6.4, height))
    y_positions = list(range(len(categories)))
    bar_height = 0.46

    for y, average, lowest, highest in zip(y_positions, average_values, lowest_values, highest_values):
        highest_color = ratio_color(highest)
        lowest_color = ratio_color(lowest)
        average_color = ratio_color(average)

        ax.barh(
            y + bar_height / 4,
            highest,
            height=bar_height / 2,
            color=rgba(highest_color, 0.28),
            edgecolor="none",
        )
        ax.barh(
            y - bar_height / 4,
            lowest,
            height=bar_height / 2,
            color=rgba(lowest_color, 0.28),
            edgecolor="none",
        )
        ax.barh(
            y,
            average,
            height=bar_height,
            facecolor="none",
            edgecolor=average_color,
            linewidth=1.8,
        )

    ax.axvspan(0.8, 1.2, color=GREEN, alpha=0.08, linewidth=0)
    ax.axvspan(0.5, 0.8, color=YELLOW, alpha=0.08, linewidth=0)
    ax.axvspan(1.2, 1.5, color=YELLOW, alpha=0.08, linewidth=0)
    ax.axvline(1.0, color="#57606a", linestyle="--", linewidth=1.0)
    ax.axvline(1.5, color=RED, linestyle=":", linewidth=1.0)

    ax.set_yticks(y_positions, categories)
    ax.invert_yaxis()
    ax.set_xlabel("calibrated multiplier vs baseline")
    ax.set_title("Benchmark Multiplication Factors")
    ax.set_xlim(0, max(1.6, max(highest_values) * 1.20))
    ax.grid(axis="x", alpha=0.18)
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=OUTLINE, linewidth=1.8, label="average"),
            Patch(facecolor=rgba(OUTLINE, 0.28), edgecolor="none", label="highest: top half"),
            Patch(facecolor=rgba(OUTLINE, 0.14), edgecolor="none", label="lowest: bottom half"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
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


def chart_url(stats: dict[str, dict[str, float]]) -> str:
    categories = list(stats)
    lowest = [round(stats[category]["lowest"], 3) for category in categories]
    average = [round(stats[category]["average"], 3) for category in categories]
    highest = [round(stats[category]["highest"], 3) for category in categories]
    max_value = max(highest)

    config = {
        "type": "bar",
        "data": {
            "labels": categories,
            "datasets": [
                {
                    "label": "highest: top half",
                    "data": highest,
                    "backgroundColor": [css_rgba(ratio_color(stats[category]["highest"]), 0.28) for category in categories],
                    "borderWidth": 0,
                    "barThickness": 10,
                },
                {
                    "label": "average",
                    "data": average,
                    "backgroundColor": "rgba(0,0,0,0)",
                    "borderColor": [ratio_color(stats[category]["average"]) for category in categories],
                    "borderWidth": 2,
                    "barThickness": 22,
                },
                {
                    "label": "lowest: bottom half",
                    "data": lowest,
                    "backgroundColor": [css_rgba(ratio_color(stats[category]["lowest"]), 0.28) for category in categories],
                    "borderWidth": 0,
                    "barThickness": 10,
                },
            ],
        },
        "options": {
            "indexAxis": "y",
            "plugins": {
                "legend": {"display": True, "position": "bottom"},
                "title": {"display": True, "text": "Benchmark Multiplication Factors"},
            },
            "scales": {
                "x": {
                    "min": 0,
                    "max": max(1.6, max_value * 1.20),
                    "title": {"display": True, "text": "calibrated multiplier vs baseline"},
                    "grid": {"color": "rgba(36, 41, 47, 0.12)"},
                },
                "y": {"grid": {"display": False}},
            },
        },
    }
    encoded = quote(json.dumps(config, separators=(",", ":")))
    height = max(240, 150 + 54 * len(categories))
    return f"https://quickchart.io/chart?width=720&height={height}&version=3&backgroundColor=white&c={encoded}"


def write_markdown(ratios: list[BenchmarkRatio], stats: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged = outside_green(ratios)

    lines = [
        "<!-- generalized-adcs-benchmark-report -->",
        "## Benchmark Report",
        "",
        f"![Benchmark multiplication factors]({chart_url(stats)})",
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
    write_markdown(ratios, stats, args.markdown)
    write_summary_json(ratios, args.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
