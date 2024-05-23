import engine
import sys

# Call the NAS engine for multiple runs. Random seed is incremented for each run.

GRAMMAR_FILE = "lenet.grammar"                  # grammar to use
CONFIG_FILE = "config_stepper_adaptive.json"    # config file to use

FIRST_RUN = 0      # first run number and random seed number to execute (zero-based, usually 0)
LAST_RUN = 9      # last run number and random seed number to execute (usually 9 or 19)

# engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(FIRST_RUN, LAST_RUN + 1):
	print(f"***** Starting {CONFIG_FILE} run #{run:02d} of {LAST_RUN}, random seed {100 + run} **********")
	engine.do_nas_search(config_file=f'config/{CONFIG_FILE}', grammar_file=f'config/{GRAMMAR_FILE}', run_number=run, override_random_seed=100 + run)

sys.exit(0)