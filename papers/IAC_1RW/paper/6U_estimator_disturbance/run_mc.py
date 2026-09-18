"""Run any estimator / disturbance 6U MC configuration (10 runs by default)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import KP, KD, TUNED_3MTQ_0RW_KD, TUNED_3MTQ_0RW_KP, run_monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", type=int, choices=(0, 1, 3), required=True, help="Number of reaction wheels.")
    parser.add_argument("--allocator", choices=("lp", "qp"), required=True)
    parser.add_argument("--goal-type", choices=("quaternion", "vector"), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    kp, kd = (TUNED_3MTQ_0RW_KP, TUNED_3MTQ_0RW_KD) if args.architecture == 0 else (KP, KD)
    run_monte_carlo(number_rw=args.architecture, allocator=args.allocator,
                    label=f"3 MTQ + {args.architecture} RW (disturbed, {args.goal_type}, {args.allocator.upper()})",
                    goal_type=args.goal_type, use_estimator=True, disturbances=True,
                    campaign_prefix="6u_estimator_disturbance", output_dir=Path(__file__).resolve().parent / "outputs",
                    default_kp=kp, default_kd=kd)


if __name__ == "__main__":
    main()
