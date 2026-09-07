#!/bin/zsh
# 3+0 and 3+3 context cells at n=100 (seeds 0-99, settled bus, one orbit, PD, both
# tasks). Final-review optional item: at n=100 the 3+3 Wilson interval no longer
# overlaps 3+1's, and the per-trial pkls give the context rows intervals + hold metrics.
#
# ONE JOB AT A TIME (standing rule): this script WAITS until the box is free -- no
# other campaign generator, no tune_planner lock, no foreign heavy python jobs, load
# settled -- and only then launches. Safe to leave running in the background.
cd /Users/patrickmckeen/ADCS_wt/iac-1rw
OUT=papers/IAC_1RW/output_data
LOG=$OUT/A_context_n100.log
PY=/Users/patrickmckeen/Documents/Generalized_ADCS/venv/bin/python

busy() {
  pgrep -f "closed_loop_nonlinear.py" >/dev/null && return 0
  pgrep -f "generate_[A-Z]_" >/dev/null && return 0
  pgrep -f "tune_planner.py" >/dev/null && return 0
  [[ -e $OUT/.tune_lock ]] && kill -0 "$(cat $OUT/.tune_lock 2>/dev/null)" 2>/dev/null && return 0
  local l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  (( ${l1%.*} >= 3 )) && return 0
  return 1
}

echo "watcher started $(date)" >> $LOG
# require the box to be free on 3 consecutive 60 s polls before launching
free_polls=0
while (( free_polls < 3 )); do
  if busy; then free_polls=0; else free_polls=$((free_polls + 1)); fi
  sleep 60
done
echo "box free; launching $(date)" >> $LOG

A_SCALE=paper A_CELLS=pd A_ONLY_NRW=0,3 A_CONTEXT_N=100 \
  caffeinate -i $PY -u papers/IAC_1RW/generate_A_baseline.py >> $LOG 2>&1
echo "context n=100 finished rc=$? $(date)" >> $LOG
echo "context n=100 finished $(date)" >> $OUT/chain.log
