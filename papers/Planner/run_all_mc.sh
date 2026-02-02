#!/bin/bash
cd /home/pmckeen/Generalized_ADCS/papers/Planner
source ../../venv/bin/activate

echo "=== Starting MC Tests at $(date) ==="

for f in generate_mc_3mtq+0rw_plan_full.py \
         generate_mc_3mtq+0rw_plan_multi.py \
         generate_mc_3mtq+0rw_plan_reduced.py \
         generate_mc_3mtq+1rw_plan_full.py \
         generate_mc_3mtq+1rw_plan_multi.py \
         generate_mc_3mtq+1rw_plan_reduced.py; do
    echo ""
    echo "=== Running $f at $(date) ==="
    python3 "$f" 2>&1 | tee -a mc_run_log.txt
    echo "=== Completed $f at $(date) ==="
done

echo ""
echo "=== All MC Tests Complete at $(date) ==="
