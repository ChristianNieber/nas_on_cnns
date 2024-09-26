#!/bin/bash
# run NAS search from a command line. Will restart if the Python process crashes.
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
# change the log level to 0 to include info messages, or to 3 to exclude everything
export TF_CPP_MIN_LOG_LEVEL=1
# the following fixes "UnicodeEncodeError: ‘charmap’ codec can’t encode characters" problem with unusual characters
export PYTHONIOENCODING=utf-8
export PYTHONLEGACYWINDOWSSTDIO=utf-8
while true; do
  python -u main_10_runs.py
  ret_code=$?
  if [ $ret_code == 0 ] || [ $ret_code == 123 ]; then
      echo 'run_nas.sh: COMPLETED WITH RETURNCODE' $ret_code
      exit 0
  fi
  echo 'run_nas.sh: PROCESS RETURNED' $ret_code ', RESTARTING PROCESS'
  echo ''
done
