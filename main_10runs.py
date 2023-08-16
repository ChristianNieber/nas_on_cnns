import engine

EXPERIMENT_MAIN_FOLDER = "DECAY3X"
EXPERIMENT_NAME = "DECAY"
GRAMMAR_FILE = "lenet.grammar"
CONFIG_FILE = "config10.json"
FIRST_RUN = 0
LAST_RUN = 10

EXPERIMENTS_PATH = "D:/experiments"
# EXPERIMENTS_PATH = "/content/gdrive/MyDrive/experiments/"

# engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(FIRST_RUN, LAST_RUN):
	print(f"***** Starting {EXPERIMENT_MAIN_FOLDER}/{EXPERIMENT_NAME}* run {run:02d} of {LAST_RUN}, random seed {100 + run} **********")
	engine.do_nas_search(EXPERIMENTS_PATH + f"/{EXPERIMENT_NAME}",
							config_file=f'config/{CONFIG_FILE}',
							grammar_file=f'config/{GRAMMAR_FILE}',
							override_experiment_name=f"{EXPERIMENT_NAME}{run:02d}",
							override_random_seed=100 + run)