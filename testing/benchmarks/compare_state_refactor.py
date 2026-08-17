"""Build a before/after report from same-machine benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

UNAFFECTED_BENCHMARKS = {
    ("disturbances", "prop_torque"): "Prop_Disturbance.torque does not read spacecraft state.",
    ("goals", "fixed_attitude_goal_to_ref"): "Fixed_Attitude_Goal.to_ref is state-independent.",
}


def _normalized(entry: dict, result: dict) -> float:
    if "normalized_seconds" in entry:
        return float(entry["normalized_seconds"])
    return float(entry["seconds"]) / float(result.get("calibration_seconds", 1.0))


def build_report(before_dir: Path, after_dir: Path, threshold: float) -> dict:
    categories = {}
    for before_path in sorted(before_dir.glob("benchmark_*.json")):
        after_path = after_dir / before_path.name
        if not after_path.exists():
            continue
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        category = before_path.stem.removeprefix("benchmark_")
        benchmarks = {}
        for name, before_entry in before["benchmarks"].items():
            after_entry = after["benchmarks"][name]
            before_normalized = _normalized(before_entry, before)
            after_normalized = _normalized(after_entry, after)
            ratio = after_normalized / before_normalized
            exemption = UNAFFECTED_BENCHMARKS.get((category, name))
            benchmarks[name] = {
                "before_seconds": before_entry["seconds"],
                "after_seconds": after_entry["seconds"],
                "before_normalized_seconds": before_normalized,
                "after_normalized_seconds": after_normalized,
                "ratio": ratio,
                "exceeds_threshold": ratio > threshold,
                "state_refactor_affected": exemption is None,
                "acceptance_exemption": exemption,
            }
        categories[category] = {
            "before_calibration_seconds": before.get("calibration_seconds"),
            "after_calibration_seconds": after.get("calibration_seconds"),
            "benchmarks": benchmarks,
        }

    state_path = after_dir / "benchmark_state.json"
    state_microbenchmarks = None
    if state_path.exists():
        state_microbenchmarks = json.loads(state_path.read_text(encoding="utf-8"))

    affected_regressions = [
        f"{category}.{name}"
        for category, data in categories.items()
        for name, benchmark in data["benchmarks"].items()
        if benchmark["state_refactor_affected"] and benchmark["exceeds_threshold"]
    ]
    return {
        "schema_version": 1,
        "threshold": threshold,
        "comparison": "Same machine, seven-sample normalized medians",
        "affected_regressions": affected_regressions,
        "acceptance_passed": not affected_regressions,
        "categories": categories,
        "state_microbenchmarks": state_microbenchmarks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before_dir", type=Path)
    parser.add_argument("after_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=1.10)
    args = parser.parse_args()

    report = build_report(args.before_dir, args.after_dir, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
