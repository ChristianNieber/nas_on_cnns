# NAS engine main module. Call do_nas_search() to run.

from os import makedirs
import pickle
import os
from shutil import copyfile
from glob import glob
from jsmin import jsmin
from pathlib import Path
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from data_augmentation import augmentation

from runstatistics import *
from plot_statistics import DEFAULT_EXPERIMENT_PATH
from logger import *
from utils import Evaluator, Individual

from strategy_stepper import StepperGrammar, StepperStrategy, RandomSearchStrategy
from strategy_fdenser import FDENSERGrammar, FDENSERStrategy

DEBUG_CONFIGURATION = 0				# use config_debug.json default configuration file instead of config.json
LOG_DEBUG = 0						# log debug messages (about caching)
LOG_MUTATIONS = 1					# log all mutations
LOG_NEW_BEST_INDIVIDUAL = 1			# log long description of new best individual
SAVE_MODELS_TO_FILE = 0             # currently not used
SAVE_MILESTONE_GENERATIONS = 50     # save milestone every 50 generations

# turn off annoying keras log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pickle_evaluator(evaluator: Evaluator, save_path):
	""" Save the Evaluator instance to later enable resuming evolution - currently unused"""
	with open(Path('%s/evaluator.pkl' % save_path), 'wb') as handle:
		pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population_and_statistics(path, population, stat: RunStatistics):
	"""
		Save the objects (pickle) necessary to later resume evolution:
		Pickled objects:
			.population
			.statistics, containing random states: numpy and random
		Useful for resuming later
		Replaces the objects of the previous generation.

		Parameters
		----------
		path: str
			path to the json file
		population : list
			a list of Individuals
		stat: RunStatistics
			statistics to store
	"""

	stat.random_state = random.getstate()
	stat.random_state_numpy = np.random.get_state()
	with open(path + 'statistics.pkl', 'wb') as handle_statistics:
		pickle.dump(stat, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)

	with open(path + 'population.pkl', 'wb') as handle_pop:
		pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population_and_statistics_milestone(save_path, generation, population, stat: RunStatistics):
	pickle_population_and_statistics(save_path + 'gen%d_' % generation, population, stat)


def unpickle_population_and_statistics(path, resume_generation=0):
	"""
		Save the objects (pickle) necessary to later resume evolution.
		Useful for later conducting more generations.
		Replaces the objects of the previous generation.
		Returns None in case any generation has been performed yet.


		Parameters
		----------
		path: str
			path where the objects needed to resume evolution are stored.
		resume_generation : int
			0 for last, or number of milestone generation (50, 100 or 150) where computation should resume


		Returns
		-------
		last_generation : int
			idx of the last performed generation
		pickle_evaluator : Evaluator
			instance of the Evaluator class used for evaluating the individuals.
			Loaded because it has the data used for training.
		pickle_population : list
			population of the last performed generation
		pickle_random : tuple
			Random module random state
		pickle_numpy : tuple
			Numpy module random state
	"""

	gen_prefix = f"gen{resume_generation}_" if resume_generation else ''
	if os.path.isfile(path + gen_prefix + 'statistics.pkl'):
		with open(path + gen_prefix + 'statistics.pkl', 'rb') as handle_statistics:
			pickled_statistics = pickle.load(handle_statistics)
		last_generation = pickled_statistics.run_generation

		# with open(path + 'evaluator.pkl', 'rb') as handle_eval:
		# pickled_evaluator = pickle.load(handle_eval)

		with open(path + gen_prefix + 'population.pkl', 'rb') as handle_pop:
			pickled_population = pickle.load(handle_pop)

		return last_generation, pickled_population, pickled_statistics

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


def select_new_parents(population, number_of_parents, after_k_folds_evaluation=False):
	"""
		Select the parent to seed the next generation.

		Parameters
		----------
		population : list
			list of instances of Individual
		number_of_parents : int
			number of parents
		after_k_folds_evaluation : bool
			select only from candidates with k_fold_metrics

		Returns
		-------
		new_parents : list
			individual that seeds the next generation
	"""

	if after_k_folds_evaluation:
		candidates = [ind for ind in population if ind.k_fold_metrics]		# only individuals where k folds evaluation has been done
	else:
		candidates = population
	# candidates ordered by fitness
	sorted_candidates = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
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


def do_nas_search(experiments_path=DEFAULT_EXPERIMENT_PATH, run_number=0, dataset='mnist', config_file='config/config.json', grammar_file='config/lenet.grammar', override_experiment_name=None, override_random_seed=None):
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
			path to the grammar file
		override_experiment_name : bool
			overrides EXPERIMENT_NAME from config file if set
		override_random_seed : bool
			overrides RANDOM_SEED from config file if set
	"""

	if DEBUG_CONFIGURATION and config_file == 'config/config.json':
		config_file = 'config/config_debug.json'

	# load config file
	config = load_config(config_file)
	EXPERIMENT_NAME = config['EVOLUTIONARY']['experiment_name']
	if override_experiment_name is not None:
		EXPERIMENT_NAME = override_experiment_name
	RESUME = config['EVOLUTIONARY']['resume']
	RESUME_GENERATION = config['EVOLUTIONARY']['resume_generation']
	RANDOM_SEED = config['EVOLUTIONARY']['random_seed']
	if override_random_seed is not None:
		RANDOM_SEED = override_random_seed
	NUM_GENERATIONS = config['EVOLUTIONARY']['num_generations']
	INITIAL_INDIVIDUALS = config['EVOLUTIONARY']['initial_individuals']
	MY = config['EVOLUTIONARY']['my']
	LAMBDA = config['EVOLUTIONARY']['lambda']
	COMMA_STRATEGY = config['EVOLUTIONARY']['comma_strategy']
	NAS_STRATEGY = config['EVOLUTIONARY']['nas_strategy']
	INITIAL_SIGMA = config['EVOLUTIONARY']['stepper_initial_sigma'] if 'stepper_initial_sigma' in config['EVOLUTIONARY'].keys() else 0.5

	NETWORK_STRUCTURE = config['NETWORK']['network_structure']
	MACRO_STRUCTURE = config['NETWORK']['macro_structure']
	OUTPUT_STRUCTURE = config['NETWORK']['output']
	NETWORK_STRUCTURE_INIT = config['NETWORK']['network_structure_init']

	USE_EVALUATION_CACHE = config['TRAINING']['use_evaluation_cache']
	EVALUATION_CACHE_FILE = config['TRAINING']['evaluation_cache_file']
	MAX_TRAINING_TIME = config['TRAINING']['max_training_time']
	MAX_TRAINING_EPOCHS = config['TRAINING']['max_training_epochs']
	K_FOLDS = config['TRAINING']['k_folds']
	SELECT_BEST_WITH_K_FOLDS_ACCURACY = config['TRAINING']['select_best_with_k_folds_accuracy']
	TEST_INIT_SEEDS = config['TRAINING']['test_init_seeds']

	data_generator = eval(config['TRAINING']['datagen'])
	data_generator_test = eval(config['TRAINING']['datagen_test'])

	if not experiments_path.endswith('/'):
		experiments_path += '/'
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
		nas_strategy = StepperStrategy(NAS_STRATEGY == "Stepper-Adaptive", initial_sigma=INITIAL_SIGMA)
	elif NAS_STRATEGY == "Random":
		grammar = StepperGrammar(grammar_file)
		nas_strategy = RandomSearchStrategy(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, NETWORK_STRUCTURE_INIT)
	else:
		raise TypeError("nas_strategy must be 'Random', 'F-DENSER' or 'Stepper'")

	nas_strategy.set_grammar(grammar)

	best_fitness = None
	best_individual_overall = None

	population = []
	last_gen = -1
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
				os.remove(f)
			for f in glob(save_path + '#' + run_prefix + '*'):
				os.remove(f)

		stat = RunStatistics(RANDOM_SEED)
		stat.init_session()

		log_bold(f"[Experiment {EXPERIMENT_NAME} in folder {save_path} run#{run_number:02}]")

	else:
		logger_configure_overwrite(False)
		last_gen, population, stat = unpickle
		if last_gen + 1 >= NUM_GENERATIONS:
			print(f'{EXPERIMENT_NAME} run#{run_number:02} in folder {save_path} is already complete, generation {last_gen}')
			return

	# set random seeds
	if RANDOM_SEED != -1:
		random.seed(RANDOM_SEED)
		np.random.seed(RANDOM_SEED)

	# create evaluator
	evaluation_cache_path = None
	if USE_EVALUATION_CACHE:
		evaluation_cache_path = Path(save_path, EVALUATION_CACHE_FILE).resolve()
	cnn_eval = Evaluator(dataset, fitness_metric_with_size_penalty, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS,  for_k_fold_validation=K_FOLDS, calculate_fitness_with_k_folds_accuracy=SELECT_BEST_WITH_K_FOLDS_ACCURACY, test_init_seeds=TEST_INIT_SEEDS,
							evaluation_cache_path=evaluation_cache_path, experiment_name=EXPERIMENT_NAME, data_generator=data_generator, data_generator_test=data_generator_test, save_path=save_path if SAVE_MODELS_TO_FILE else None)

	if unpickle is None:
		# pickle_evaluator(cnn_eval, save_path_with_run) # save evaluator

		# set random seeds
		if RANDOM_SEED != -1:
			random.seed(RANDOM_SEED)
			np.random.seed(RANDOM_SEED)

		initial_population_size = LAMBDA if INITIAL_INDIVIDUALS == 'random' else 1
		log(f'Creating the initial population of {initial_population_size}')
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
				new_individual.evaluate_individual(grammar, cnn_eval, stat)
				if new_individual.fitness:  # new individual could be invalid, then try again
					break
				log_bold("Invalid individual created, trying again")
			population.append(new_individual)
			cnn_eval.flush_evaluation_cache()  # flush after every created individual
		log()

	# in case there is a previous population, load it
	else:
		stat.init_session()

		random.setstate(stat.random_state)
		np.random.set_state(stat.random_state_numpy)

		new_parents = select_new_parents(population, MY)
		population = new_parents

		log('\n========================================================================================================')
		log_bold(f'Resuming experiment {EXPERIMENT_NAME} run#{run_number:02} in folder {save_path} after generation {last_gen}')

	for gen in range(last_gen + 1, NUM_GENERATIONS):
		# generate offspring by mutations and evaluate population
		generation_list = []
		if gen:
			for idx in range(LAMBDA):
				parent = select_parent(population[0:MY])
				while True:
					new_individual = nas_strategy.mutation(parent, gen, idx)
					fitness = new_individual.evaluate_individual(grammar, cnn_eval, stat)
					if fitness:
						break
				generation_list.append(new_individual)

		# select candidates for parents
		population += generation_list
		if gen and COMMA_STRATEGY:
			selection_pool = generation_list
		else:
			selection_pool = population
		new_parents = select_new_parents(selection_pool, MY)

		log_nolf(f'[Generation %d in {EXPERIMENT_NAME}] ' % gen)

		# if there are new parent candidates, do K-fold validation on them
		new_parent_candidates_count = 0
		if SELECT_BEST_WITH_K_FOLDS_ACCURACY:
			for idx, parent in enumerate(new_parents):
				if not parent.is_parent:
					log_bold(f"K-folds evaluation of candidate for {'best' if idx==0 else f'rank #{idx+1}'} {parent.description()}")
					parent.evaluate_individual_k_folds(grammar, cnn_eval, K_FOLDS, stat, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
					new_parent_candidates_count += 1
			if new_parent_candidates_count:
				new_parents = select_new_parents(selection_pool, MY, after_k_folds_evaluation=True)
				count = sum(1 for parent in new_parents if not parent.is_parent)
				if count == 0:
					log_bold('New parent candidate fails K-folds evaluation')

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

					if K_FOLDS and not parent.k_fold_metrics:
						log_bold(f'*** Evaluating k folds for new parent {parent.description()} ***')
						parent.evaluate_individual_k_folds(grammar, cnn_eval, K_FOLDS, stat, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
					if TEST_INIT_SEEDS:
						parent.k_fold_metrics = None
						log_bold(f'*** Evaluating init folds for new parent {parent.description()} ***')
						parent.evaluate_individual_init_seeds(grammar, cnn_eval, TEST_INIT_SEEDS, stat, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)

				if best_fitness is None or parent.fitness > best_fitness:
					if best_fitness:
						log_bold(f'*** New best individual replaces {population[0].description()} ***')
					if LOG_NEW_BEST_INDIVIDUAL:
						parent.log_long_description("New best" if best_fitness else "Initial")
					else:
						log_bold(f'*** New best: {parent.description()}')

					best_fitness = parent.fitness
					best_individual_overall = parent

					# copy new best individual's weights
					if SAVE_MODELS_TO_FILE and os.path.isfile(parent.model_save_path):
						copyfile(parent.model_save_path, save_path_with_run + 'best.h5')
					with open(save_path_with_run + 'best_parent.pkl', 'wb') as handle:
						pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

				elif not COMMA_STRATEGY:
					log_bold(f'New individual rank {idx} replaces {population[idx].description()}')
					log_bold(f'New rank {idx}: {parent.description()}')

		# flush evaluation cache after every generation
		cnn_eval.flush_evaluation_cache()

		# remove temporary files to free disk space
		if gen > 1 and SAVE_MODELS_TO_FILE:
			for x in range(LAMBDA):
				individual_path = Path(save_path_with_run, 'individual-%d-%d.h5' % (gen - 2, x))
				if os.path.isfile(individual_path):
					os.remove(individual_path)

		best_individual = new_parents[0]

		while len(generation_list) < LAMBDA:			# extend gen 0 population for easier statistics
			generation_list.append(population[0])
		best_in_generation_idx = np.argmax([ind.fitness for ind in generation_list])
		best_in_generation = generation_list[best_in_generation_idx]
		if NAS_STRATEGY == "Stepper":
			best_in_generation.record_stepwidth_statistics(stat.stepwidth_stats)
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
		pickle_population_and_statistics(save_path_with_run, population, stat)
		if (gen + 1) % SAVE_MILESTONE_GENERATIONS == 0:
			pickle_population_and_statistics_milestone(save_path_with_run, gen + 1, population, stat)

		generation_list.clear()
		selection_pool.clear()

		# keep only best as new parents
		population = new_parents
		for parent in population:
			parent.is_parent = True

	parent = population[0]
	parent.log_long_description('Final Individual')

	if len(stat.k_fold_accuracy_stds) > 1:
		log(f"Average folds accuracy SD: {average_standard_deviation(stat.k_fold_accuracy_stds):.5f} {stat.k_fold_accuracy_stds}")
		log(f"Average folds final accuracy SD: {average_standard_deviation(stat.k_fold_final_accuracy_stds):.5f} {stat.k_fold_final_accuracy_stds}")

	# if NAS_STRATEGY == "F-DENSER":
	# 	nas_strategy.dump_mutated_variables(stat.evaluations_total)


def test_saved_model(save_path, name='best.h5'):
	# datagen_test = ImageDataGenerator()
	# evaluator = Evaluator('mnist', fitness_function)

	if not save_path.endswith('/'):
		save_path += '/'
	with open(Path(save_path, 'evaluator.pkl'), 'rb') as f_data:
		evaluator = pickle.load(f_data)
	x_final_test = evaluator.dataset['x_final_test']
	y_final_test = evaluator.dataset['y_final_test']

	model = load_model(Path(save_path, name))
	accuracy = Evaluator.test_model_with_data(model, x_final_test, y_final_test)
	log('Best test accuracy: %f' % accuracy)
	return model
