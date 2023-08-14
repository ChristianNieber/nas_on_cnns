import engine

EXPERIMENT_NAME = "DECAY"
EXPERIMENT_SUFFIX = "256"
GRAMMAR_FILE = "lenet256.grammar"
FDENSER = False
START_RUN = 0
EXPERIMENTS_PATH = "D:/experiments"

# engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(START_RUN, 10):
	print(f"***** Starting {EXPERIMENT_NAME}{EXPERIMENT_SUFFIX} run {run:02d} of 10, random seed {100 + run} **********")
	engine.do_nas_search(EXPERIMENTS_PATH + f"/{EXPERIMENT_NAME}{EXPERIMENT_SUFFIX}X",
							config_file=('config/config_fdenser.json' if FDENSER else 'config/config10.json'),
							grammar_file=f'config/{GRAMMAR_FILE}',
							override_experiment_name=f"{EXPERIMENT_NAME}{run:02d}",
							override_random_seed=100 + run)
