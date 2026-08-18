#!/bin/zsh
# Fires Campaign C the moment Campaign A's PD half releases the machine.
# C is independent of A's results, so it launches even if A ends with an error --
# a crashed A costs its own cells, not the night.
while pgrep -f "generate_A_baseline" >/dev/null 2>&1; do sleep 120; done
cd /Users/patrickmckeen/ADCS_wt/iac-1rw
C_SCALE=paper /Users/patrickmckeen/Documents/Generalized_ADCS/venv/bin/python -u \
  papers/IAC_1RW/generate_C_bias.py > papers/IAC_1RW/output_data/C_run.log 2>&1
echo "C finished $(date)" >> papers/IAC_1RW/output_data/chain.log
