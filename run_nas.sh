#!/bin/bash
# run NAS search from a command line. Will restart if the Python process crashes.
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
# change the log level to 0 to include info messages, or to 3 to exclude everything
export TF_CPP_MIN_LOG_LEVEL=2
# the following fixes "UnicodeEncodeError: ‘charmap’ codec can’t encode characters" problem with unusual characters
export PYTHONIOENCODING=utf-8
export PYTHONLEGACYWINDOWSSTDIO=utf-8
while true; do
  python -u main_10_runs.py
  # process regularly returns 0 for completion, 2 for regular interruption/restart
  ret_code=$?
  if [ $ret_code == 0 ]; then
      echo 'run_nas.sh: COMPLETED WITH RETURNCODE' $ret_code
      exit 0
  elif [ $ret_code == 2 ]; then
    unset NAS_RERANDOMIZE_AFTER_CRASH
    crash_counter=0
    echo 'run_nas.sh: NAS Process returned ' $ret_code ', RESTARTING PROCESS'
  else
    # observed returncodes for crashes: 1 for access violation, 134 for aborted, 139 for signal 11:SIGSEGV
    export NAS_RERANDOMIZE_AFTER_CRASH=$((++crash_counter))
    echo 'run_nas.sh: NAS Process returned ' $ret_code ', setting NAS_RERANDOMIZE_AFTER_CRASH =' $NAS_RERANDOMIZE_AFTER_CRASH
  fi
  echo ''
done
