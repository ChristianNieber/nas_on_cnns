import engine
import sys

# Call the NAS engine for multiple runs. Random seed is incremented for each run.

# List of config files to run
CONFIG_FILE_LIST = [
	"config_stepper_adaptive.json",
	"config_stepper_decay.json",
	"config_fdenser.json",
	"config_random_search.json"
]

FIRST_RUN = 0      # first run number and random seed number to execute (zero-based, usually 0)
LAST_RUN = 20      # last run number and random seed number to execute (usually 9 or 19)

RETURN_AFTER_GENERATIONS = 48    # To minimize memory leaks and GPU memory fragmentation, restart the process every n generations when running from run_nas.sh

for config_file in CONFIG_FILE_LIST:
	for run in range(FIRST_RUN, LAST_RUN + 1):
		# print(f"***** Starting {config_file} run #{run:02d} of {LAST_RUN}, random seed {100 + run} **********")
		is_completed = engine.do_nas_search(run_number=run,
											config_file=f'config/{config_file}',
											override_random_seed=100 + run,
											return_after_generations=RETURN_AFTER_GENERATIONS)
		if not is_completed:
			sys.exit(2)
	print (f"[{config_file} {LAST_RUN+1} runs completed]")

print(f"[All complete]")
sys.exit(0)
