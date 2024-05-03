import engine
import sys

# Call the NAS engine for multiple runs. Random seed is incremented for each run.

EXPERIMENT_MAIN_FOLDER = "ADAPTIVE_test"     # Main folder name. Numbered subfolders will be created within.
EXPERIMENT_NAME = "ADAPTIVE"                 # Template for the numbered subfolder names. This will create subfolders e.g. ADAPTIVE00 to ADAPTIVE19
GRAMMAR_FILE = "lenet.grammar"               # grammar to use
CONFIG_FILE = "config_stepper_decay.json"    # config file to use


# EXPERIMENT_MAIN_FOLDER = "FDENSER20"
# EXPERIMENT_NAME = "FDENSER"
# GRAMMAR_FILE = "lenet_fdenser.grammar"    # FDENSER code needs its own grammar
# CONFIG_FILE = "config_fdenser.json"

FIRST_RUN = 0      # first run number and random seed number to execute (zero-based, usually 0)
LAST_RUN = 9      # last run number and random seed number to execute (usually 9 or 19)

EXPERIMENTS_PATH = "D:/experiments"     # Path where main folder will be created
# EXPERIMENTS_PATH = "/content/gdrive/MyDrive/experiments"

# engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(FIRST_RUN, LAST_RUN + 1):
	print(f"***** Starting {EXPERIMENT_MAIN_FOLDER}/{EXPERIMENT_NAME}* run {run:02d} of {LAST_RUN}, random seed {100 + run} **********")
	engine.do_nas_search(EXPERIMENTS_PATH + f"/{EXPERIMENT_MAIN_FOLDER}",
							config_file=f'config/{CONFIG_FILE}',
							grammar_file=f'config/{GRAMMAR_FILE}',
							override_experiment_name=f"{EXPERIMENT_NAME}{run:02d}",
							override_random_seed=100 + run)

sys.exit(0)