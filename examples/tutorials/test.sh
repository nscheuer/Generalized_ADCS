#!/bin/bash

TUTORIALS_DIR="$HOME/Github_Projects/Generalized_ADCS/examples/tutorials"
VENV="$HOME/Github_Projects/Generalized_ADCS/venv/bin/activate"
SCRIPTS=("01_underactuated_control.py" "02_noisy_control.py" "03_simple_estimation.py" "04_complex_estimation.py" "05_orbit_estimation.py")
BRANCHES=("Numba_Optimizations" "main")
RUNS="${RUNS:-10}"
LOG_FILE="${LOG_FILE:-benchmark_results.log}"
TUTORIAL_TIMEOUT="${TUTORIAL_TIMEOUT:-30m}"

if [ -n "$SCRIPT_LIST" ]; then
    read -r -a SCRIPTS <<<"$SCRIPT_LIST"
fi

export MPLBACKEND=Agg

source "$VENV"
cd "$TUTORIALS_DIR"

echo "Starting benchmarks — $(date)" | tee "$LOG_FILE"
echo "Runs per branch: $RUNS" | tee -a "$LOG_FILE"
echo "Per-tutorial timeout: $TUTORIAL_TIMEOUT" | tee -a "$LOG_FILE"

declare -A SUM_SECONDS
declare -A COUNT_OK
declare -A COUNT_FAIL
declare -A COUNT_TIMEOUT
declare -A BRANCH_AVG

for BRANCH in "${BRANCHES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "Branch: $BRANCH" | tee -a "$LOG_FILE"

    if ! git switch "$BRANCH" 2>&1 | tee -a "$LOG_FILE"; then
        echo "Failed to switch to $BRANCH. Stopping benchmark." | tee -a "$LOG_FILE"
        exit 1
    fi

    for RUN in $(seq 1 $RUNS); do
        echo "Run $RUN/$RUNS on $BRANCH" | tee -a "$LOG_FILE"

        for SCRIPT in "${SCRIPTS[@]}"; do
            echo "  Timing $SCRIPT" | tee -a "$LOG_FILE"
            OUTPUT_FILE=$(mktemp)
            START_NS=$(date +%s%N)
            if timeout --signal=SIGINT --kill-after=30s "$TUTORIAL_TIMEOUT" python3 -u - "$SCRIPT" >"$OUTPUT_FILE" 2>&1 <<'PY'
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import runpy
import sys

plt.show = lambda *args, **kwargs: None

script = sys.argv[1]
sys.argv = [script]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    plt.close("all")
PY
            then
                STATUS=0
            else
                STATUS=$?
            fi
            END_NS=$(date +%s%N)

            ELAPSED_S=$(awk -v start="$START_NS" -v end="$END_NS" 'BEGIN { printf "%.3f", (end-start)/1000000000 }')

            KEY="$BRANCH|$SCRIPT"

            if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ]; then
                COUNT_TIMEOUT["$KEY"]=$(( ${COUNT_TIMEOUT["$KEY"]:-0} + 1 ))
                COUNT_FAIL["$KEY"]=$(( ${COUNT_FAIL["$KEY"]:-0} + 1 ))
                echo "    TIMEOUT after $TUTORIAL_TIMEOUT" | tee -a "$LOG_FILE"
                grep -E "Error|Traceback|Exception" "$OUTPUT_FILE" | head -n 12 | tee -a "$LOG_FILE"
            elif [ "$STATUS" -ne 0 ]; then
                COUNT_FAIL["$KEY"]=$(( ${COUNT_FAIL["$KEY"]:-0} + 1 ))
                echo "    FAILED with status $STATUS" | tee -a "$LOG_FILE"
                grep -E "Error|Traceback|Exception" "$OUTPUT_FILE" | head -n 12 | tee -a "$LOG_FILE"
            else
                SUM_SECONDS["$KEY"]=$(awk -v a="${SUM_SECONDS["$KEY"]:-0}" -v b="$ELAPSED_S" 'BEGIN { printf "%.6f", a+b }')
                COUNT_OK["$KEY"]=$(( ${COUNT_OK["$KEY"]:-0} + 1 ))
            fi

            rm -f "$OUTPUT_FILE"
        done
    done

    echo "" | tee -a "$LOG_FILE"
    echo "Summary for branch: $BRANCH" | tee -a "$LOG_FILE"
    BRANCH_SUM=0
    BRANCH_COUNT=0
    for SCRIPT in "${SCRIPTS[@]}"; do
        KEY="$BRANCH|$SCRIPT"
        OK=${COUNT_OK["$KEY"]:-0}
        FAIL=${COUNT_FAIL["$KEY"]:-0}
        TIMEOUTS=${COUNT_TIMEOUT["$KEY"]:-0}
        if [ "$OK" -gt 0 ]; then
            AVG=$(awk -v s="${SUM_SECONDS["$KEY"]:-0}" -v c="$OK" 'BEGIN { printf "%.3f", s/c }')
            BRANCH_SUM=$(awk -v a="$BRANCH_SUM" -v b="${SUM_SECONDS["$KEY"]:-0}" 'BEGIN { printf "%.6f", a+b }')
            BRANCH_COUNT=$(( BRANCH_COUNT + OK ))
        else
            AVG="N/A"
        fi
        echo "  $SCRIPT -> avg: $AVG s, success: $OK, fail: $FAIL, timeout: $TIMEOUTS" | tee -a "$LOG_FILE"
    done

    if [ "$BRANCH_COUNT" -gt 0 ]; then
        BRANCH_AVG["$BRANCH"]=$(awk -v s="$BRANCH_SUM" -v c="$BRANCH_COUNT" 'BEGIN { printf "%.3f", s/c }')
        echo "  Overall average for $BRANCH: ${BRANCH_AVG["$BRANCH"]} s" | tee -a "$LOG_FILE"
    else
        BRANCH_AVG["$BRANCH"]="N/A"
        echo "  Overall average for $BRANCH: N/A (no successful runs)" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "OVERALL:" | tee -a "$LOG_FILE"

NUMBA_AVG="${BRANCH_AVG["Numba_Optimizations"]:-N/A}"
MAIN_AVG="${BRANCH_AVG["main"]:-N/A}"
echo "Numba_Optimizations overall avg: $NUMBA_AVG s" | tee -a "$LOG_FILE"
echo "main overall avg: $MAIN_AVG s" | tee -a "$LOG_FILE"

if [ "$NUMBA_AVG" != "N/A" ] && [ "$MAIN_AVG" != "N/A" ]; then
    DELTA=$(awk -v m="$MAIN_AVG" -v n="$NUMBA_AVG" 'BEGIN { printf "%.3f", m-n }')
    SPEEDUP=$(awk -v m="$MAIN_AVG" -v n="$NUMBA_AVG" 'BEGIN { if (n>0) printf "%.3f", m/n; else print "inf" }')
    PCT=$(awk -v m="$MAIN_AVG" -v n="$NUMBA_AVG" 'BEGIN { if (m>0) printf "%.2f", (m-n)*100/m; else print "0.00" }')
    echo "Delta (main - Numba_Optimizations): $DELTA s" | tee -a "$LOG_FILE"
    echo "Speedup factor (main / Numba_Optimizations): ${SPEEDUP}x" | tee -a "$LOG_FILE"
    echo "Percent faster vs main: ${PCT}%" | tee -a "$LOG_FILE"
fi

echo "Benchmarks complete — $(date)" | tee -a "$LOG_FILE"
echo "Results saved to $LOG_FILE"