import numpy as np
import random
from copy import deepcopy
from time import time
from os import makedirs
import pickle
import os
from shutil import copyfile
from glob import glob
import json
from jsmin import jsmin
from pathlib import Path
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from data_augmentation import augmentation
from grammar import Grammar, mutation_dsge

from utils import Evaluator, Individual, stat
from logger import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEBUG_CONFIGURATION = 0				# use config_debug.json default configuration file instead of config.json
LOG_DEBUG = 0						# log debug messages (for caching)
LOG_MUTATIONS = 1					# log all mutations
LOG_NEW_BEST_INDIVIDUAL = 1			# log long description of new best individual

# global variables set from config.json
USE_NETWORK_SIZE_PENALTY = 0
PENALTY_PARAMETERS_TARGET = 0

def fitness_metric_with_size_penalty(accuracy, parameters):
	if USE_NETWORK_SIZE_PENALTY:
		return 3 - (((1.0 - accuracy)/0.02) ** 2 + parameters / PENALTY_PARAMETERS_TARGET)
	else:
		return accuracy

def save_population_statistics(population, save_path, gen):
	"""
		Save the current population statistics in json.
		For each individual:
			.name: name as <generation>-<index>
			.phenotype: phenotype of the individual
			.fitness: fitness of the individual
			.metrics: other evaluation metrics (e.g., loss, accuracy)
			.metrics.parameters: number of network trainable parameters
			.metrics.training_epochs: number of performed training epochs
			.metrics.training_time: time (sec) the network took to perform training_epochs
			.million_inferences_time: measured time per million inferences

		Parameters
		----------
		population : list
			list of Individual instances
		save_path : str
			path to the json file
		gen : int
			current generation
	"""

	json_dump = []

	for ind in population:
		json_dump.append(ind.json_statistics())

	with open(Path('%s/gen_%d.json' % (save_path, gen)), 'w') as f_json:
		f_json.write(json.dumps(json_dump, indent=4))


def pickle_evaluator(evaluator, save_path):
	"""
		Save the Evaluator instance to later enable resuming evolution

		Parameters
		----------
		evaluator: Evaluator
			instance of the Evaluator class
		save_path: str
			path to the json file
	"""

	with open(Path('%s/evaluator.pkl' % save_path), 'wb') as handle:
		pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population(gen, population, save_path):
	"""
		Save the objects (pickle) necessary to later resume evolution:
		Pickled objects:
			.population
			.random states: numpy and random
		Useful for later conducting more generations.
		Replaces the objects of the previous generation.

		Parameters
		----------
		gen : int
			generation
		population : list
			a list of Individuals
		save_path: str
			path to the json file
	"""

	path = save_path + 'gen_%d_' % gen
	with open(path + 'population.pkl', 'wb') as handle_pop:
		pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

	with open(path + 'random.pkl', 'wb') as handle_random:
		pickle.dump(random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

	with open(path + 'numpy.pkl', 'wb') as handle_numpy:
		pickle.dump(np.random.get_state(), handle_numpy, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_statistics(stat, save_path):
	with open(save_path + 'statistics.pkl', 'wb') as handle_statistics:
		pickle.dump(stat, handle_statistics, protocol=pickle.HIGHEST_PROTOCOL)


def get_total_epochs(save_path, last_gen):
	"""
		Compute the total number of performed epochs.

		Parameters
		----------
		save_path: str
			path where the objects needed to resume evolution are stored.
		last_gen : int
			count the number of performed epochs until the last_gen generation

		Returns
		-------
		total_epochs : int
			sum of the number of epochs performed by all trainings
	"""

	total_epochs = 0
	for gen in range(0, last_gen + 1):
		with open(Path('%s/gen_%d.json' % (save_path, gen))) as json_file:
			j = json.load(json_file)
			training_epochs = [elm['training_epochs'] for elm in j]
			total_epochs += sum(training_epochs)

	return total_epochs


def unpickle_population(save_path):
	"""
		Save the objects (pickle) necessary to later resume evolution.
		Useful for later conducting more generations.
		Replaces the objects of the previous generation.
		Returns None in case any generation has been performed yet.


		Parameters
		----------
		save_path: str
			path where the objects needed to resume evolution are stored.


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

	if os.path.isfile(save_path + 'statistics.pkl'):
		with open(save_path + 'statistics.pkl', 'rb') as handle_statistics:
			pickle_statistics = pickle.load(handle_statistics)
		last_generation = pickle_statistics.run_generation

		path = save_path + 'gen_%d_' % last_generation

		with open(save_path + 'evaluator.pkl', 'rb') as handle_eval:
			pickled_evaluator = pickle.load(handle_eval)

		with open(path + 'population.pkl', 'rb') as handle_pop:
			pickled_population = pickle.load(handle_pop)

		with open(path + 'random.pkl', 'rb') as handle_random:
			pickle_random = pickle.load(handle_random)

		with open(path + 'numpy.pkl', 'rb') as handle_numpy:
			pickle_numpy = pickle.load(handle_numpy)

		return last_generation, pickled_evaluator, pickled_population, pickle_random, pickle_numpy, pickle_statistics

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

	# Get best individual just according to fitness
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

		Returns
		-------
		new_parents : list
			individual that seeds the next generation
			:param after_k_folds_evaluation:
	"""

	if after_k_folds_evaluation:
		candidates = [ind for ind in population if ind.k_fold_metrics]		# only individuals where k folds evaluation has been done
	else:
		candidates = population
	# candidates ordered by fitness
	sorted_candidates = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
	return sorted_candidates[0:number_of_parents]


def mutation(parent, grammar, add_layer, re_use_layer, remove_layer, add_connection, remove_connection, dsge_layer, macro_layer, gen=0, idx=0):
	"""
		Network mutations: add and remove layer, add and remove connections, macro structure


		Parameters
		----------
		parent : Individual
			individual to be mutated

		grammar : Grammar
			Grammar instance, used to perform the initialisation and the genotype
			to phenotype
		gen : int
			Generation count
		idx : int
			index in generation
		add_layer : float
			add layer mutation rate
		re_use_layer : float
			when adding a new layer, defines the mutation rate of using an already
			existing layer, i.e., copy by reference
		remove_layer : float
			remove layer mutation rate
		add_connection : float
			add connection mutation rate
		remove_connection : float
			remove connection mutation rate
		dsge_layer : float
			inner lever genotype mutation rate
		macro_layer : float
			inner level of the macro layers (i.e., learning, data-augmentation) mutation rate

		Returns
		-------
		ind : Individual
			mutated individual
	"""

	# deep copy parent
	ind = deepcopy(parent)
	ind.parent_id = parent.id

	# name for new individual
	ind.id = f"{gen}-{idx}"

	# mutation resets training results
	ind.reset_training()

	for module_idx, module in enumerate(ind.modules):
		# add-layer (duplicate or new)
		for _ in range(random.randint(1, 2)):
			if len(module.layers) < module.max_expansions and random.random() <= add_layer:
				insert_pos = random.randint(0, len(module.layers))
				if random.random() <= re_use_layer and len(module.layers):
					source_layer_index = random.randint(0, len(module.layers)-1)
					new_layer = module.layers[source_layer_index]
					layer_phenotype = grammar.decode_layer(module.module_name, new_layer)
					ind.log_mutation(f"copy layer {module_idx}#{insert_pos}/{len(module.layers)} from {source_layer_index} - {layer_phenotype}")
				else:
					new_layer = grammar.initialise_layer(module.module_name)
					layer_phenotype = grammar.decode_layer(module.module_name, new_layer)
					ind.log_mutation(f"insert layer {module_idx}#{insert_pos}/{len(module.layers)} - {layer_phenotype}")

				# fix connections
				for _key_ in sorted(module.connections, reverse=True):
					if _key_ >= insert_pos:
						for value_idx, value in enumerate(module.connections[_key_]):
							if value >= insert_pos - 1:
								module.connections[_key_][value_idx] += 1

						module.connections[_key_ + 1] = module.connections.pop(_key_)

				module.layers.insert(insert_pos, new_layer)

				# make connections of the new layer
				if insert_pos == 0:
					module.connections[insert_pos] = [-1]
				else:
					connection_possibilities = list(range(max(0, insert_pos - module.levels_back), insert_pos - 1))
					if len(connection_possibilities) < module.levels_back - 1:
						connection_possibilities.append(-1)

					sample_size = random.randint(0, len(connection_possibilities))

					module.connections[insert_pos] = [insert_pos - 1]
					if sample_size > 0:
						module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)

		# remove-layer
		for _ in range(random.randint(1, 2)):
			if len(module.layers) > module.min_expansions and random.random() <= remove_layer:
				remove_idx = random.randint(0, len(module.layers) - 1)
				layer_phenotype = grammar.decode_layer(module.module_name, module.layers[remove_idx])
				ind.log_mutation(f"remove layer {module_idx}#{remove_idx}/{len(module.layers)} - {layer_phenotype}")
				del module.layers[remove_idx]

				# fix connections
				for _key_ in sorted(module.connections):
					if _key_ > remove_idx:
						if _key_ > remove_idx + 1 and remove_idx in module.connections[_key_]:
							module.connections[_key_].remove(remove_idx)

						for value_idx, value in enumerate(module.connections[_key_]):
							if value >= remove_idx:
								module.connections[_key_][value_idx] -= 1
						module.connections[_key_ - 1] = list(set(module.connections.pop(_key_)))

				if remove_idx == 0:
					module.connections[0] = [-1]

		for layer_idx, layer in enumerate(module.layers):
			# dsge mutation
			if random.random() <= dsge_layer:
				mutation_dsge(ind, layer, grammar, f"{module_idx}#{layer_idx}")

			# add connection
			if layer_idx != 0 and random.random() <= add_connection:
				connection_possibilities = list(range(max(0, layer_idx - module.levels_back), layer_idx - 1))
				connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
				if len(connection_possibilities) > 0:
					module.connections[layer_idx].append(random.choice(connection_possibilities))

			# remove connection
			r_value = random.random()
			if layer_idx != 0 and r_value <= remove_connection:
				connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx - 1]))
				if len(connection_possibilities) > 0:
					r_connection = random.choice(connection_possibilities)
					module.connections[layer_idx].remove(r_connection)

	# macro level mutation
	for macro_idx, macro in enumerate(ind.macro):
		if random.random() <= macro_layer:
			mutation_dsge(ind, macro, grammar, "learning")

	return ind


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


def do_nas_search(experiments_directory='../Experiments/', dataset='mnist', config_file='config/config.json', grammar_file='config/lenet.grammar'):
	"""
		(1+lambda)-ES

		Parameters
		----------
		experiments_directory : str
			directory where all experiments are saved
		dataset : str
			dataset to be solved
		config_file : str
			path to the configuration file
		grammar_file : str
			path to the grammar file
	"""

	global stat
	global USE_NETWORK_SIZE_PENALTY
	global PENALTY_PARAMETERS_TARGET

	if DEBUG_CONFIGURATION and config_file=='config/config.json':
		config_file = 'config/config_debug.json'

	# load config file
	config = load_config(config_file)
	EXPERIMENT_NAME = config['EVOLUTIONARY']['experiment_name']
	RESUME = config['EVOLUTIONARY']['resume']
	RANDOM_SEED = config['EVOLUTIONARY']['random_seed']
	NUM_GENERATIONS = config['EVOLUTIONARY']['num_generations']
	INITIAL_POPULATION_SIZE = config['EVOLUTIONARY']['initial_population_size']
	INITIAL_INDIVIDUALS = config['EVOLUTIONARY']['initial_individuals']
	MY = config['EVOLUTIONARY']['my']
	LAMBDA = config['EVOLUTIONARY']['lambda']
	REUSE_LAYER = config['EVOLUTIONARY']['MUTATIONS']['reuse_layer']
	ADD_LAYER = config['EVOLUTIONARY']['MUTATIONS']['add_layer']
	REMOVE_LAYER = config['EVOLUTIONARY']['MUTATIONS']['remove_layer']
	ADD_CONNECTION = config['EVOLUTIONARY']['MUTATIONS']['add_connection']
	REMOVE_CONNECTION = config['EVOLUTIONARY']['MUTATIONS']['remove_connection']
	DSGE_LAYER = config['EVOLUTIONARY']['MUTATIONS']['dsge_layer']
	MACRO_LAYER = config['EVOLUTIONARY']['MUTATIONS']['macro_layer']

	NETWORK_STRUCTURE = config['NETWORK']['network_structure']
	MACRO_STRUCTURE = config['NETWORK']['macro_structure']
	OUTPUT_STRUCTURE = config['NETWORK']['output']
	NETWORK_STRUCTURE_INIT = config['NETWORK']['network_structure_init']
	LEVELS_BACK = config["NETWORK"]["levels_back"]

	USE_EVALUATION_CACHE = config['TRAINING']['use_evaluation_cache']
	EVALUATION_CACHE_FILE = config['TRAINING']['evaluation_cache_file']
	MAX_TRAINING_TIME = config['TRAINING']['max_training_time']
	MAX_TRAINING_EPOCHS = config['TRAINING']['max_training_epochs']
	USE_NETWORK_SIZE_PENALTY = config['TRAINING']['use_network_size_penalty']
	PENALTY_PARAMETERS_TARGET = config['TRAINING']['penalty_connections_target']
	RETEST_BEST_WITH_FINAL_TEST_SET = config['TRAINING']['retest_best_with_final_test_set']
	REEVALUATE_BEST_WITH_K_FOLDS = config['TRAINING']['reevaluate_best_with_k_folds']
	data_generator = eval(config['TRAINING']['datagen'])
	data_generator_test = eval(config['TRAINING']['datagen_test'])

	if not experiments_directory.endswith('/'):
		experiments_directory += '/'
	save_path = experiments_directory + EXPERIMENT_NAME + '/'

	log_file_path = save_path + '#' + EXPERIMENT_NAME + '.log'
	init_logger(log_file_path, overwrite=True)
	logger_configuration(logger_log_training=True, logger_log_mutations=LOG_MUTATIONS, logger_log_debug=LOG_DEBUG)

	# load grammar
	grammar = Grammar(grammar_file)

	best_fitness = None

	# load previous population content (if any)
	unpickle = unpickle_population(save_path) if RESUME else None

	# if there is not a previous population
	if unpickle is None:
		# create directories
		makedirs(save_path, exist_ok=True)

		# delete old files
		if not RESUME:
			for f in glob(str(Path(save_path, '*'))):
				os.remove(f)

		# copy config files to path
		copyfile(config_file, save_path + Path(config_file).name)
		copyfile(grammar_file, save_path + Path(grammar_file).name)

		# stat = RunStatistics()

		# set random seeds
		if RANDOM_SEED != -1:
			random.seed(RANDOM_SEED)
			np.random.seed(RANDOM_SEED)

		# create evaluator
		evaluation_cache_path = None
		if USE_EVALUATION_CACHE:
			evaluation_cache_path=Path(save_path, EVALUATION_CACHE_FILE).resolve()
		cnn_eval = Evaluator(dataset, fitness_metric_with_size_penalty, for_k_fold_validation=REEVALUATE_BEST_WITH_K_FOLDS, evaluation_cache_path=evaluation_cache_path, experiment_name=EXPERIMENT_NAME)

		# save evaluator
		pickle_evaluator(cnn_eval, save_path)

		last_gen = -1


		log(f'[Experiment {EXPERIMENT_NAME}] Creating the initial population of {INITIAL_POPULATION_SIZE}')
		population = []
		for idx in range(INITIAL_POPULATION_SIZE):
			new_individual = Individual(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, 0, idx)
			if INITIAL_INDIVIDUALS == "lenet":
				new_individual.initialise_as_lenet(grammar)
			elif INITIAL_INDIVIDUALS == "perceptron":
				new_individual.initialise_as_perceptron(grammar)
			elif INITIAL_INDIVIDUALS == "random":
				new_individual.initialise_individual_random(grammar, LEVELS_BACK, REUSE_LAYER, NETWORK_STRUCTURE_INIT)
			else:
				raise RuntimeError(f"invalid value '{INITIAL_INDIVIDUALS}' of initial_individuals")
			new_individual.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
			cnn_eval.flush_evaluation_cache()
			population.append(new_individual)
		log()

	# in case there is a previous population, load it
	else:
		init_logger(log_file_path, overwrite=False)

		last_gen, cnn_eval, population, pkl_random, pkl_numpy, stat = unpickle
		random.setstate(pkl_random)
		np.random.set_state(pkl_numpy)
		new_parents = select_new_parents(population, MY)
		population = new_parents
		log('\n========================================================================================================')
		log(f'[Experiment {EXPERIMENT_NAME}] Resuming evaluation after generation {last_gen}:')

	stat.init_session()

	for gen in range(last_gen + 1, NUM_GENERATIONS):
		# generate offspring by mutations and evaluate population
		generation_list = []
		if gen:
			for idx in range(LAMBDA):
				parent = select_parent(population[0:MY])
				while True:
					new_individual = mutation(parent, grammar, ADD_LAYER, REUSE_LAYER, REMOVE_LAYER, ADD_CONNECTION, REMOVE_CONNECTION, DSGE_LAYER, MACRO_LAYER, gen, idx)
					fitness = new_individual.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
					if fitness:
						break
				generation_list.append(new_individual)

		# select candidates for parents
		population += generation_list
		new_parents = select_new_parents(population, MY)

		log_nolf('[Gen %d] ' % gen)

		# if there are new parent candidates, do K-fold validation on them
		new_parent_candidates_count = 0
		if REEVALUATE_BEST_WITH_K_FOLDS:
			for idx, parent in enumerate(new_parents):
				if not parent.is_parent:
					log_bold(f"K-folds evaluation of candidate for {'best' if idx==0 else f'rank #{idx+1}'} {parent.short_description()}")
					start_time = time()
					parent.evaluate_individual_k_folds(grammar, cnn_eval, REEVALUATE_BEST_WITH_K_FOLDS, data_generator, data_generator_test, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
					new_parent_candidates_count += 1
					# flush caache after every individual
					cnn_eval.flush_evaluation_cache()
			if new_parent_candidates_count:
				new_parents = select_new_parents(population, MY, after_k_folds_evaluation=True)
				count = sum(1 for parent in new_parents if not parent.is_parent)
				if count == 0:
					log_bold('New parent candidate fails K-folds evaluation')

		for idx, parent in enumerate(new_parents):
			if not parent.is_parent:
				parent.is_parent = True
				if RETEST_BEST_WITH_FINAL_TEST_SET and not parent.metrics.final_test_accuracy:
					parent.calculate_final_test_accuracy(cnn_eval)
				if parent.parent_id:
					original_parent = next(ind for ind in population if ind.id == parent.parent_id)
					parent.log_mutation_summary(f"{parent.id} new #{idx+1}: [{parent.short_description()}] <- [{original_parent.short_description()}] Δfitness={parent.fitness - original_parent.fitness:.5f} Δacc={parent.metrics.accuracy - original_parent.metrics.accuracy:.5f}")
				if best_fitness == None or parent.fitness > best_fitness:
					if best_fitness:
						log_bold(f'*** New best individual replaces {population[0].short_description()} ***')
					if LOG_NEW_BEST_INDIVIDUAL:
						parent.log_long_description("New best" if best_fitness else "Initial")
					else:
						log_bold(f'*** New best: {parent.short_description()}')

					best_fitness = parent.fitness

					# copy new best individual's weights
					if os.path.isfile(parent.model_save_path):
						copyfile(parent.model_save_path, Path(save_path, 'best.h5'))
					with open('%s/best_parent.pkl' % save_path, 'wb') as handle:
						pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)
				else:
					log_bold(f'New individual rank {idx+1} replaces {population[idx].short_description()}')
					log_bold(f'New rank {idx+1}: {parent.short_description()}')

		# flush evaluation cache after every generation
		cnn_eval.flush_evaluation_cache()

		# remove temporary files to free disk space
		if gen > 1:
			for x in range(LAMBDA):
				individual_path = Path(save_path, 'individual-%d-%d.h5' % (gen - 2, x))
				if os.path.isfile(individual_path):
					os.remove(individual_path)

		best_individual = new_parents[0]

		while len(generation_list) < LAMBDA:			# extend gen 0 population for easier statistics
			generation_list.append(population[0])
		best_in_generation_idx = np.argmax([ind.fitness for ind in generation_list])
		best_in_generation = generation_list[best_in_generation_idx]
		log(f'Best: {best_individual.short_description()}, in generation: {best_in_generation.short_description()}')
		for idx in range(1, len(new_parents)):
			log(f'  #{idx+1}: {new_parents[idx].short_description()}')

		stat.record_best(best_individual)
		assert len(generation_list) == LAMBDA
		stat.record_generation(generation_list)

		generation_list.clear()

		# save population
		save_population_statistics(population, save_path, gen)
		pickle_population(gen, population, save_path)
		pickle_statistics(stat, save_path)
		# stat.save_to_json_file(save_path)

		# keep only best as new parents
		population = new_parents
		for parent in population:
			parent.is_parent = True

	parent = population[0]
	parent.log_long_description('Final Individual')


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
