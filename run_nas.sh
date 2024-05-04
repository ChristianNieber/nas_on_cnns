export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
while true; do
  python -u main_20_runs.py
  ret_code=$?
  echo 'PROCESS RETURNED' $ret_code
  if [ $ret_code == 0 ] || [ $ret_code == 123 ]; then
      exit 0
  fi
  echo 'RESTARTING PROCESS'
done
