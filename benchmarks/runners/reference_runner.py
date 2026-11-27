"""Reference solutions (run in any environment with NumPy and SciPy).

Prints {"<t>": final-state dict} for each requested final time ``t`` (seconds),
integrating the shared scenario with DOP853 at rtol = 1e-13.
"""

from __future__ import annotations

import json
import sys

from common import S


def main():
    times = sorted({float(t) for t in sys.argv[1:]})
    out = {}
    for t in times:
        steps = round(t / S.DT)
        ref = S.reference_solution(steps)
        ref["dcm_bi"] = S.quat_b_from_i_to_dcm(ref["q_b_from_i"])
        out[repr(t)] = ref
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
