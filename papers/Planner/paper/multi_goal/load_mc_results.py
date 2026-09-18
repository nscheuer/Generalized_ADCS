"""Load and validate the sequential multi-goal Planner campaign.

The campaign's historical result files are expected in ``outputs/``.  The
loader accepts either the JSON summaries emitted by ``generate_p2.3_multigoal``
or native ``.sim`` files once those are available.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# Requested paper-table values: Config, Goal, Acq.%, Held.%, old, t_acq [s], SS [deg].
EXPECTED = {
    ("3+1", "A"): (96, 77, 10, 232, 1.28),
    ("3+1", "B"): (99, 99, 14, 48, 0.52),
    ("3+1", "C"): (100, 100, 98, 37, 0.29),
    ("3+3", "A"): (100, 100, 22, 156, 0.43),
    ("3+3", "B"): (100, 100, 25, 19, 0.31),
    ("3+3", "C"): (100, 100, 100, 23, 0.17),
}


def print_expected() -> None:
    print("Config Goal Acq.% Held% old tacq [s] SS [°]")
    for (config, goal), values in EXPECTED.items():
        acq, held, old, tacq, ss = values
        print(f"{config:<6} {goal:<4} {acq:>5} {held:>5} {old:>3} {tacq:>8} {ss:>5.2f}")


def check_json(path: Path) -> None:
    """Check the available JSON summary against its aggregate acquisition data."""
    payload = json.loads(path.read_text())
    config = payload["config"]
    aggregate = payload["aggregate"]
    print(f"\nChecking {path.name} ({payload.get('n_trials', '?')} trials)")
    for goal in ("A", "B", "C"):
        if goal not in aggregate:
            raise KeyError(f"{path}: missing aggregate goal {goal}")
        row = aggregate[goal]
        expected = EXPECTED[(config, goal)]
        actual = (round(row["acquired_pct"]), round(row["conv_pct"]))
        print(f"  {config} {goal}: acquired={actual[0]}% held={actual[1]}%", end="")
        if actual == expected[:2]:
            print(" [MATCH]")
        else:
            print(f" [MISMATCH; expected {expected[:2]}%]")
        print("    Note: this JSON schema does not contain t_acq or the requested SS metric.")


def main() -> None:
    print_expected()
    json_files = sorted(OUTPUTS.glob("P2.3_multigoal_*.json"))
    sim_files = sorted(OUTPUTS.glob("P2.3_multigoal_*.sim"))
    if not json_files and not sim_files:
        raise FileNotFoundError(
            f"No multi-goal results found in {OUTPUTS}. Expected P2.3_multigoal_*.json or .sim files."
        )
    for path in json_files:
        check_json(path)
    if sim_files:
        print("\nNative .sim files:")
        for path in sim_files:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
