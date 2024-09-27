import engine
import sys
from os import environ


EXPERIMENTS_PATH = "~/nas/experiments.fashion4"

FIRST_RUN = 0      # first run number and random seed number to execute (zero-based, usually 0)
LAST_RUN = 39      # last run number and random seed number to execute (usually 9 or 19)
DATASET = 'fashion-mnist'
GRAMMAR_FILE = 'lenet.grammar'
MAX_TRAINING_EPOCHS = 10
MAX_TRAINING_TIME = 10 if DATASET == 'mnist' else 50

# List of config files to run
CONFIG_FILE_LIST = [
	"config_stepper_adaptive.json",
	"config_stepper_decay.json",
	"config_fdenser.json",
	"config_random_search.json"
]

RETURN_AFTER_GENERATIONS = 20    # To minimize memory leaks and GPU memory fragmentation, restart the process every n generations when running from run_nas.sh

rerandomize_after_crash=0
if 'NAS_RERANDOMIZE_AFTER_CRASH' in environ:
	rerandomize_after_crash=int(environ['NAS_RERANDOMIZE_AFTER_CRASH'])
	print(f"***Reading NAS_RERANDOMIZE_AFTER_CRASH={rerandomize_after_crash} from environment***")

# Call the NAS engine for multiple config files and multiple runs. Random seed is incremented for each run.
for config_file in CONFIG_FILE_LIST:
	for run in range(FIRST_RUN, LAST_RUN + 1):
		is_completed = engine.do_nas_search(experiments_path=EXPERIMENTS_PATH,
											run_number=run,
											dataset=DATASET,
											config_file=f'config/{config_file}',
											grammar_file=GRAMMAR_FILE,
											override_random_seed=100 + run,
											override_max_training_epochs=MAX_TRAINING_EPOCHS,
											override_max_training_time=MAX_TRAINING_TIME,
											override_evaluation_cache_file=f"../cache_{DATASET}_{MAX_TRAINING_EPOCHS}ep_{MAX_TRAINING_TIME}s.pkl",
											reevaluate_best_10_seeds=True,
											return_after_generations=RETURN_AFTER_GENERATIONS,
											rerandomize_after_crash=rerandomize_after_crash)
		if not is_completed:
			sys.exit(2)
	print(f"[{config_file} {LAST_RUN+1} runs completed]")

print(f"[All complete]")
sys.exit(0)
