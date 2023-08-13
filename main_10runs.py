import engine

EXPERIMENT_NAME = "DECAY"
FDENSER = False
EXPERIMENTS_PATH = "D:/experiments"

# engine.do_nas_search(experiments_path, config_file='config/config.json', grammar_file='config/lenet.grammar')

for run in range(4, 10):
	print(f"***** Starting {EXPERIMENT_NAME}{run:02d} of 10, random seed {100 + run} **********")
	engine.do_nas_search(EXPERIMENTS_PATH + f"/{EXPERIMENT_NAME}10", config_file=('config/config_fdenser.json' if FDENSER else 'config/config10.json'), grammar_file='config/lenet.grammar', override_experiment_name=f"{EXPERIMENT_NAME}{run:02d}", override_random_seed=100 + run)
