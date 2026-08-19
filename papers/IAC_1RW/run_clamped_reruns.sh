#!/bin/zsh
# Clamped reruns, sequential: 1rw cells of A, then all of C. Runs under one caffeinate.
cd /Users/patrickmckeen/ADCS_wt/iac-1rw
A_SCALE=paper A_CELLS=pd A_ONLY_NRW=1 \
  /Users/patrickmckeen/Documents/Generalized_ADCS/venv/bin/python -u \
  papers/IAC_1RW/generate_A_baseline.py > papers/IAC_1RW/output_data/A_1rw_clamped.log 2>&1
C_SCALE=paper \
  /Users/patrickmckeen/Documents/Generalized_ADCS/venv/bin/python -u \
  papers/IAC_1RW/generate_C_bias.py > papers/IAC_1RW/output_data/C_clamped.log 2>&1
echo "clamped reruns finished $(date)" >> papers/IAC_1RW/output_data/chain.log
