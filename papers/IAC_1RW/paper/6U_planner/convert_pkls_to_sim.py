"""Aggregate Planner trial pickles into a SimulationResults-compatible .sim archive."""

from __future__ import annotations

import argparse
from datetime import datetime
import lzma
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE / "trials_10"
OUTPUT_DIR = HERE / "outputs"
DEFAULT_OUTPUT_NAME = "6u_3mtq_1rw_boresight_planner_mc_10"


def convert_trial(path: Path) -> tuple[dict, dict, int]:
    with path.open("rb") as stream:
        data = pickle.load(stream)

    run = {
        "satellite": None,
        "est_satellite": SimpleNamespace(
            number_RW=1,
            act_bias_len=0,
            att_sens_bias_len=0,
            dist_param_len=0,
        ),
        "time_s": np.asarray(data["time"]),
        "state_hist": np.asarray(data["state"]),
        "est_state_hist": np.asarray(data["est"]),
        "control_hist": np.asarray(data["u"]),
    }
    standard_keys = {"run_id", "config", "time", "state", "est", "u"}
    config = {
        "source_pickle": path.name,
        "config": data.get("config", {}),
        "diagnostics": {
            key: value for key, value in data.items() if key not in standard_keys
        },
    }
    return run, config, int(data["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("1rw_reduced_planner_seed*.pkl"))
    if not paths:
        raise RuntimeError(f"no Planner pickles found in {args.input_dir}")
    converted = [convert_trial(path) for path in paths]
    payload = {
        "schema_version": 2,
        "runs": [run for run, _, _ in converted],
        "configs": [config for _, config, _ in converted],
        "run_ids": [run_id for _, _, run_id in converted],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{args.output_name}_{datetime.now():%Y%m%d_%H%M%S}.sim"
    with lzma.open(output, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(converted)} runs to {output}")


if __name__ == "__main__":
    main()
