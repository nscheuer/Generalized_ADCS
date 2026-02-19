#!/bin/bash
cd /home/pmckeen/Generalized_ADCS
source venv/bin/activate

configs=(
    "generate_mc_3mtq+1rw_plan_full.py"
    "generate_mc_3mtq+1rw_plan_reduced.py"
    "generate_mc_3mtq+1rw_plan_multi.py"
    "generate_mc_3mtq+0rw_plan_full.py"
    "generate_mc_3mtq+0rw_plan_reduced.py"
    "generate_mc_3mtq+0rw_plan_multi.py"
)

for cfg in "${configs[@]}"; do
    echo "=== Starting $cfg at $(date) ==="
    timeout 7200 python "papers/Planner/$cfg" -n 100
    echo "=== Finished $cfg at $(date) ==="
    echo ""
done
echo "ALL DONE at $(date)"
