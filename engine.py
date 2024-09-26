# NAS engine main module. Call do_nas_search() to run.

import pickle
from os import makedirs, remove, path
from shutil import copyfile
from glob import glob
from jsmin import jsmin
from pathlib import Path
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from data_augmentation import augmentation

from runstatistics import *
from plot_statistics import DEFAULT_EXPERIMENT_PATH, fixup_path
from logger import *
from utils import Evaluator, Individual

from strategy_stepper import StepperGrammar, StepperStrategy, RandomSearchStrategy
from strategy_fdenser import FDENSERGrammar, FDENSERStrategy

LOG_DEBUG = 1						    # log debug messages (about caching)
LOG_MUTATIONS = 0					    # log all mutations
LOG_NEW_BEST_INDIVIDUAL = 0			    # log long description of new best individual
SAVE_MILESTONE_GENERATIONS = 50         # save milestone every 50 generations
EXPERIMENTAL_MULTITHREADING = False     # use multithreading when multiple (physical or logical) GPUS are available

# turn off Keras log messages - does not work?
# from os import environ
# environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

total_completed_generations = 0         # keeps track of completed generations over all runs


def pickle_population_and_statistics(save_path, stat: RunStatistics, population, best_individuals_list=None):
	"""
		Save the objects (pickle) necessary to later resume evolution:
		Pickled objects:
			statistics, containing random states: numpy and random
			population
			best individuals list

		Parameters
		----------
		save_path: str
			path to the json file
		stat: RunStatistics
			statistics to store
		population : list
			a list of Individuals
		best_individuals_list : None or list of individuals
	"""

	stat.random_state = random.getstate()
	stat.random_state_numpy = np.random.get_state()
	with open(save_path + 'statistics.pkl', 'wb') as handle_statistics:
		pickle.dump(stat, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)

	with open(save_path + 'population.pkl', 'wb') as handle_pop:
		pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

	if best_individuals_list:
		with open(save_path + 'best_individuals.pkl', 'wb') as handle_statistics:
			pickle.dump(best_individuals_list, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population_and_statistics_milestone(save_path, generation, stat: RunStatistics, population, best_individuals_list):
	pickle_population_and_statistics(save_path + 'gen%d_' % generation, stat, population, best_individuals_list)


def pickle_best_individuals(save_path, best_individuals_list):
	with open(save_path + 'best_individuals.pkl', 'wb') as handle_statistics:
		pickle.dump(best_individuals_list, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_population_and_statistics(load_path, resume_generation=0):
	"""
		Save the objects (pickle) necessary to later resume evolution.
		Useful for later conducting more generations.
		Replaces the objects of the previous generation.
		Returns None in case any generation has been performed yet.


		Parameters
		----------
		load_path: str
			path where the objects needed to resume evolution are stored.
		resume_generation : int
			0 for last, or number of milestone generation (50, 100 or 150) where computation should resume


		Returns
		-------
		last_generation : int
			idx of the last performed generation
		pickled_statistics : RunStatistics
			statistics
		pickled_population : list of Individual
			population of the last performed generation
		pickled_best_individuals_list : list of Individual
			list of best individuals (empty in old experiments)
	"""

	gen_prefix = f"gen{resume_generation}_" if resume_generation else ''
	if path.isfile(load_path + gen_prefix + 'statistics.pkl'):
		with open(load_path + gen_prefix + 'statistics.pkl', 'rb') as handle_statistics:
			pickled_statistics = pickle.load(handle_statistics)
		last_generation = pickled_statistics.run_generation

		with open(load_path + gen_prefix + 'population.pkl', 'rb') as handle_pop:
			pickled_population = pickle.load(handle_pop)

		pickled_best_individuals_list = []
		best_individuals_file = load_path + gen_prefix + 'best_individuals.pkl'
		if path.isfile(best_individuals_file):
			with open(best_individuals_file, 'rb') as handle_best_individuals:
				pickled_best_individuals_list = pickle.load(handle_best_individuals)

		return last_generation, pickled_statistics, pickled_population, pickled_best_individuals_list

	else:
		return None


def select_parent(parents):
	"""
		Select the parent to seed the next generation.

		Parameters
		----------
		parents : list
			list of potential parent Individuals

		Returns
		-------
		parent : Individual
			individual that seeds the next generation
	"""

	# Get the best individual just according to fitness
	idx = random.randint(0, len(parents) - 1)
	return parents[idx]


def select_new_parents(population, number_of_parents):
	"""
		Select the parent to seed the next generation.

		Parameters
		----------
		population : list
			list of instances of Individual
		number_of_parents : int
			number of parents

		Returns
		-------
		new_parents : list
			individual that seeds the next generation
	"""
	# candidates ordered by fitness
	sorted_candidates = sorted(population, key=lambda ind: ind.fitness, reverse=True)
	return sorted_candidates[0:number_of_parents]


def load_config(config_file):
	"""
		Load configuration json file.

		Parameters
		----------
		config_file : str
			path to the configuration file

		Returns
		-------
		config : dict
			configuration json file
	"""

	with open(Path(config_file)) as js_file:
		minified = jsmin(js_file.read())

	config = json.loads(minified)
	return config


def evaluate_generation(generation_list: list[Individual], grammar: StepperGrammar, cnn_eval: Evaluator, stat: RunStatistics, force_reevaluation=False):
	"""
		Evaluate a whole generation of individuals
		Will only evaluate individuals where metrics is None, unless force_reevaluation is True.
	"""
	parallel_list = []
	for ind in generation_list:
		if force_reevaluation:
			ind.metrics = None
		if ind.metrics is None:
			parallel_list.append(ind)
			# if len(parallel_list) >= cnn_eval.get_n_gpus():
			# 	if not Individual.evaluate_multiple_individuals(parallel_list, grammar, cnn_eval, stat):
			# 		return False
			# 	parallel_list = []

	return Individual.evaluate_multiple_individuals(parallel_list, grammar, cnn_eval, stat)


def do_nas_search(experiments_path=DEFAULT_EXPERIMENT_PATH, run_number=0, dataset='mnist', config_file='config/config.json', grammar_file='lenet.grammar',
					override_experiment_name=None, override_random_seed=None, override_evaluation_cache_file=None, override_max_training_time=None, override_max_training_epochs=None,
					reevaluate_best_10_seeds=False, return_after_generations=0):
	"""
		do (my+/,lambda)-ES for NAS search

		Parameters
		----------
		experiments_path : str
			directory where all experiments are saved
		run_number : int
			number of run (0-based)
		dataset : str
			dataset to be solved
		config_file : str
			path to the configuration file
		grammar_file : str
			name of the grammar file (searched in same path as config file)
		override_experiment_name : str
			overrides experiment_name from config file if set
		override_random_seed : int
			overrides random_seed from config file if set
		override_evaluation_cache_file : str
			overrides evaluation_cache_file from config file if set
		override_max_training_time : int
			overrides max_training_time from config file if set
		override_max_training_epochs : int
			overrides max_training_epochs from config file if set
		reevaluate_best_10_seeds : bool
			reevaluate average accuracy of new best with 10 folds
		return_after_generations : int
			0 or max number of completed generations after which the function returns

		returns bool:
			True - completed search
			False - not completed because max_generations were process
	"""
	global total_completed_generations

	config = load_config(config_file)
	EXPERIMENT_NAME = config['EVOLUTIONARY']['experiment_name']
	if override_experiment_name is not None:
		EXPERIMENT_NAME = override_experiment_name
	RESUME = config['EVOLUTIONARY']['resume']
	RESUME_GENERATION = config['EVOLUTIONARY']['resume_generation']
	RANDOM_SEED = override_random_seed if override_random_seed is not None else config['EVOLUTIONARY']['random_seed']
	NUM_GENERATIONS = config['EVOLUTIONARY']['num_generations']
	INITIAL_INDIVIDUALS = config['EVOLUTIONARY']['initial_individuals']
	MY = config['EVOLUTIONARY']['my']
	LAMBDA = config['EVOLUTIONARY']['lambda']
	COMMA_STRATEGY = config['EVOLUTIONARY']['comma_strategy']
	NAS_STRATEGY = config['EVOLUTIONARY']['nas_strategy']
	grammar_file = config['EVOLUTIONARY']['grammar'] if 'grammar' in config['EVOLUTIONARY'].keys() else grammar_file
	i = config_file.rfind('/')
	if i:
		grammar_file = config_file[:i+1] + grammar_file
	MAX_PARAMETERS = config['EVOLUTIONARY']['max_parameters'] if 'max_parameters' in config['EVOLUTIONARY'].keys() else 0
	STEPPER_INITIAL_SIGMA = config['EVOLUTIONARY']['stepper_initial_sigma'] if 'stepper_initial_sigma' in config['EVOLUTIONARY'].keys() else 0.5

	NETWORK_STRUCTURE = config['NETWORK']['network_structure']
	MACRO_STRUCTURE = config['NETWORK']['macro_structure']
	OUTPUT_STRUCTURE = config['NETWORK']['output']
	NETWORK_STRUCTURE_INIT = config['NETWORK']['network_structure_init']

	USE_EVALUATION_CACHE = config['TRAINING']['use_evaluation_cache']
	EVALUATION_CACHE_FILE = override_evaluation_cache_file if override_evaluation_cache_file else config['TRAINING']['evaluation_cache_file']

	OVERRIDE_BATCH_SIZE = config['TRAINING']['batch_size'] if 'batch_size' in config['TRAINING'].keys() else None

	MAX_TRAINING_TIME = override_max_training_time if override_max_training_time else config['TRAINING']['max_training_time']
	MAX_TRAINING_EPOCHS = override_max_training_epochs if override_max_training_epochs else config['TRAINING']['max_training_epochs']

	USE_DATA_AUGMENTATION = config['TRAINING']['use_data_augmentation']

	experiments_path = fixup_path(experiments_path)
	if not Path(experiments_path).exists():
		log_bold(f"Creating experiments directory {experiments_path}")
		makedirs(experiments_path)         # create directory
	save_path = experiments_path + EXPERIMENT_NAME + '/'
	run_prefix = f"r{run_number:02}_"
	save_path_with_run = save_path + run_prefix

	log_file_path = save_path + '#' + run_prefix + ' ' + EXPERIMENT_NAME + '.log'
	init_logger(log_file_path)
	logger_configuration(logger_log_training=True, logger_log_mutations=LOG_MUTATIONS, logger_log_debug=LOG_DEBUG)

	# load grammar and init strategy
	if NAS_STRATEGY == "F-DENSER":
		grammar = FDENSERGrammar(grammar_file)
		nas_strategy = FDENSERStrategy()
	elif NAS_STRATEGY == "Stepper-Decay" or NAS_STRATEGY == "Stepper-Adaptive":
		grammar = StepperGrammar(grammar_file)
		nas_strategy = StepperStrategy(NAS_STRATEGY == "Stepper-Adaptive", initial_sigma=STEPPER_INITIAL_SIGMA)
	elif NAS_STRATEGY == "Random":
		grammar = StepperGrammar(grammar_file)
		nas_strategy = RandomSearchStrategy(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, NETWORK_STRUCTURE_INIT)
	else:
		raise TypeError("nas_strategy must be 'Random', 'F-DENSER', 'Stepper-Adaptive', or 'Stepper-Decay'")

	nas_strategy.set_grammar(grammar)

	best_fitness = None
	best_individual_overall = None
	best_individuals_list = []
	save_best_individuals_list = False

	population = []
	last_gen = -1
	generation_time = 0
	# load previous population content (if any)
	unpickle = unpickle_population_and_statistics(save_path_with_run, RESUME_GENERATION) if RESUME else None
	# if there is not a previous population/statistics file
	if unpickle is None:
		if not Path(save_path).exists():
			makedirs(save_path)         # create directory
			copyfile(config_file, save_path + Path(config_file).name)       # copy config files to path for documentation
			copyfile(grammar_file, save_path + Path(grammar_file).name)
		# delete old files
		if not RESUME:
			for f in glob(save_path_with_run + '*'):
				remove(f)
			for f in glob(save_path + '#' + run_prefix + '*'):
				remove(f)

		stat = RunStatistics(RANDOM_SEED, run_nas_strategy=NAS_STRATEGY, run_number=run_number, run_dataset=dataset)
		stat.init_session()

	else:
		logger_configure_overwrite(False)
		last_gen, stat, population, best_individuals_list = unpickle
		if last_gen + 1 >= NUM_GENERATIONS:
			print(f'{EXPERIMENT_NAME} run#{run_number:02} is already complete ({last_gen+1} generations)')
			return True

	# set random seeds
	if RANDOM_SEED != -1:
		random.seed(RANDOM_SEED)
		np.random.seed(RANDOM_SEED)

	# create evaluator
	evaluation_cache_path = None
	if USE_EVALUATION_CACHE:
		evaluation_cache_path = Path(save_path, EVALUATION_CACHE_FILE).resolve()
	cnn_eval = Evaluator(dataset, fitness_metric_with_size_penalty, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS, max_parameters=MAX_PARAMETERS,
							evaluation_cache_path=evaluation_cache_path, experiment_name=EXPERIMENT_NAME, use_augmentation=USE_DATA_AUGMENTATION,
							override_batch_size=OVERRIDE_BATCH_SIZE, n_gpus=0)

	if unpickle is None:
		# set random seeds
		if RANDOM_SEED != -1:
			random.seed(RANDOM_SEED)
			np.random.seed(RANDOM_SEED)

		initial_population_size = LAMBDA if INITIAL_INDIVIDUALS == 'random' else 1
		start_time = time()
		log_bold(f"[{EXPERIMENT_NAME} run#{run_number:02} in folder '{save_path}', {RANDOM_SEED=}: Creating the initial population of {initial_population_size}]")
		for idx in range(initial_population_size):
			while True:
				new_individual = Individual(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, 0, idx)
				if INITIAL_INDIVIDUALS == "lenet":
					new_individual.initialise_as_lenet(grammar)
				elif INITIAL_INDIVIDUALS == "perceptron":
					new_individual.initialise_as_perceptron(grammar)
				elif INITIAL_INDIVIDUALS == "random":
					new_individual.initialise_random(grammar, NETWORK_STRUCTURE_INIT)
				else:
					raise RuntimeError(f"invalid value '{INITIAL_INDIVIDUALS}' of initial_individuals")
				if new_individual.evaluate_individual(grammar, cnn_eval, stat):     # new individual could be invalid, then try again
					break
				log_bold("Invalid individual created, trying again")
			population.append(new_individual)
			cnn_eval.flush_evaluation_cache()  # flush after every created individual

		generation_time = time() - start_time
		stat.eval_time_of_generation = [generation_time]

	# in case there is a previous population, load it
	else:
		stat.init_session()

		random.setstate(stat.random_state)
		np.random.set_state(stat.random_state_numpy)

		new_parents = select_new_parents(population, MY)
		population = new_parents

		log_bold(f"[Resuming experiment {EXPERIMENT_NAME} run#{run_number:02} at generation {last_gen + 1} in folder '{save_path}']")

	generation_list = []
	for gen in range(last_gen + 1, NUM_GENERATIONS):
		# generate offspring by mutations and evaluate population
		if gen:
			generation_start_time = time()
			if EXPERIMENTAL_MULTITHREADING:
				idx = 0
				while True:
					while len(generation_list) < LAMBDA:
						parent = select_parent(population[0:MY])
						while True:
							new_individual = nas_strategy.mutation(parent, gen, idx)
							if new_individual.validate_individual(grammar, cnn_eval, stat, use_cache=True):
								break
						idx += 1
						generation_list.append(new_individual)
					evaluate_generation(generation_list, grammar, cnn_eval, stat)
					generation_list = [ind for ind in generation_list if ind.metrics]   # delete individuals without metrics. This indicates an exception occurred.
					if len(generation_list) >= LAMBDA:                                  # fill up with new individuals if any were deleted
						break
					log_bold('Individual evaluation in generation caused exception, filling up with new individual')
			else:
				for idx in range(LAMBDA):
					parent = select_parent(population[0:MY])
					while True:
						new_individual = nas_strategy.mutation(parent, gen, idx)
						if new_individual.evaluate_individual(grammar, cnn_eval, stat, use_cache=False):
							break
					generation_list.append(new_individual)
			generation_time = time() - generation_start_time
			stat.eval_time_of_generation.append(generation_time)
		# select candidates for parents
		population += generation_list
		if gen and COMMA_STRATEGY:
			selection_pool = generation_list
		else:
			selection_pool = population
		new_parents = select_new_parents(selection_pool, MY)

		log_nolf(f"[Generation {gen} in {EXPERIMENT_NAME} run#{run_number:02} took {generation_time:.2f} s] ")

		for idx, parent in enumerate(new_parents):
			if not parent.is_parent:
				parent.is_parent = True
				if parent.parent_id:
					orig_parent_list = [ind for ind in population if ind.id == parent.parent_id]
					if len(orig_parent_list) == 1:
						orig_parent = orig_parent_list[0]
						parent.log_mutation_summary(f"{parent.id} new #{idx+1}: [{parent.description()}] <- [{orig_parent.description()}] d_fitness={parent.fitness - orig_parent.fitness:.5f} d_acc={parent.metrics.accuracy - orig_parent.metrics.accuracy:.5f}")
					else:
						log_warning(f"parent {parent.parent_id} of {parent.id} not found in current population!")

				if best_fitness is None or parent.fitness > best_fitness:
					if best_fitness:
						log_bold(f'*** New best individual replaces {population[0].description()} ***')
					if LOG_NEW_BEST_INDIVIDUAL:
						parent.log_long_description("New best" if best_fitness else "Initial")
					else:
						log_bold(f'*** New best: {parent.description()}')
					if reevaluate_best_10_seeds:
						parent.k_fold_metrics = parent.evaluate_individual_k_folds(grammar, cnn_eval, stat, epochs=parent.metrics.training_epochs, num_folds=0, num_random_seeds=10)
					best_individuals_list.append(parent)
					save_best_individuals_list = True

					best_fitness = parent.fitness
					best_individual_overall = parent

				elif not COMMA_STRATEGY:
					log_bold(f'New individual rank {idx} replaces {population[idx].description()}')
					log_bold(f'New rank {idx}: {parent.description()}')

		# flush evaluation cache after every generation
		cnn_eval.flush_evaluation_cache()

		best_individual = new_parents[0]

		while len(generation_list) < LAMBDA:			# extend gen 0 population for easier statistics
			generation_list.append(population[0])
		best_in_generation_idx = np.argmax([ind.fitness for ind in generation_list])
		best_in_generation = generation_list[best_in_generation_idx]
		if COMMA_STRATEGY:
			best_individual_overall.record_statistics(stat.best)
			best_individual.record_statistics(stat.best_in_gen)
			log(f'Best: {best_individual.description()}, overall: {best_individual_overall.short_description()}')
		else:
			best_individual.record_statistics(stat.best)
			best_in_generation.record_statistics(stat.best_in_gen)
			log(f'Best: {best_individual.description()}, in generation: {best_in_generation.description()}')

		for idx in range(1, len(new_parents)):
			log(f'  #{idx+1}: {new_parents[idx].description()}')

		assert len(generation_list) == LAMBDA
		stat.record_generation(generation_list)

		# save population and statistics
		pickle_population_and_statistics(save_path_with_run, stat, population, best_individuals_list if save_best_individuals_list else None)
		save_best_individuals_list = False
		if (gen + 1) % SAVE_MILESTONE_GENERATIONS == 0:
			pickle_population_and_statistics_milestone(save_path_with_run, gen + 1, stat, population, best_individuals_list)

		generation_list.clear()
		selection_pool.clear()

		# keep only best as new parents
		population = new_parents
		for parent in population:
			parent.is_parent = True

		total_completed_generations += 1
		if return_after_generations and total_completed_generations >= return_after_generations and gen < NUM_GENERATIONS-1:
			log_bold(f"[Exiting after {total_completed_generations} completed generations, {EXPERIMENT_NAME} run#{run_number} after generation {gen}]")
			return False

	parent = population[0]
	parent.log_long_description('Final Individual')

	stat.log_statistics_summary(1)

	log_bold(f"[{EXPERIMENT_NAME} Run#{run_number} completed]")

	return True

def test_saved_model(save_path):
	# datagen_test = ImageDataGenerator()
	# evaluator = Evaluator('mnist', fitness_function)

	save_path = fixup_path(save_path)
	with open(Path(save_path, 'evaluator.pkl'), 'rb') as f_data:
		evaluator = pickle.load(f_data)
	# x_final_test = evaluator.dataset['x_final_test']
	# y_final_test = evaluator.dataset['y_final_test']

	# model = load_model(Path(save_path, name))
	# accuracy = Evaluator.test_model_with_data(model, x_final_test, y_final_test)
	# log('Best test accuracy: %f' % accuracy)
	# return model
