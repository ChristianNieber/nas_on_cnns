import numpy as np
import random
from operator import itemgetter
from grammar import Grammar
from utils import Evaluator, Individual, test_model_with_dataset
from copy import deepcopy
from os import makedirs
import pickle
import os
from shutil import copyfile
from glob import glob
import json
from jsmin import jsmin
from pathlib import Path
from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from data_augmentation import augmentation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

USE_NETWORK_SIZE_PENALTY = 0
PENALTY_CONNECTIONS_TARGET = 0

def fitness_metric_with_size_penalty(accuracy, trainable_parameters):
	if (USE_NETWORK_SIZE_PENALTY):
		error_measure = (1.0-accuracy)*50
		return 3 - (error_measure ** 2 + trainable_parameters / PENALTY_CONNECTIONS_TARGET)
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
			.parameters: number of network trainable parameters
			.training_epochs: number of performed training epochs
			.training_time: time (sec) the network took to perform training_epochs
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
			list of Individual instances
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
		pickle_population_fitness : list
			ordered list of fitnesses of the last population of individuals
		pickle_random : tuple
			Random random state
		pickle_numpy : tuple
			Numpy random state
	"""

	json_file_paths = glob(str(Path(save_path, '*.json')))

	if json_file_paths:
		json_file_paths = [int(path.split(os.sep)[-1].replace('gen_', '').replace('.json', '')) for path in json_file_paths]
		last_generation = max(json_file_paths)

		path = save_path + 'gen_%d_' % last_generation

		with open(save_path + 'evaluator.pkl', 'rb') as handle_eval:
			pickled_evaluator = pickle.load(handle_eval)

		with open(path + 'population.pkl', 'rb') as handle_pop:
			pickled_population = pickle.load(handle_pop)

		with open(path + 'random.pkl', 'rb') as handle_random:
			pickle_random = pickle.load(handle_random)

		with open(path + 'numpy.pkl', 'rb') as handle_numpy:
			pickle_numpy = pickle.load(handle_numpy)

		# total_epochs = get_total_epochs(save_path, last_generation)

		return last_generation, pickled_evaluator, pickled_population, pickle_random, pickle_numpy

	else:
		return None


def select_parent(parents):
	"""
		Select the parent to seed the next generation.

		Parameters
		----------
		population : list
			list of instances of Individual
		population_fits : list
			ordered list of fitnesses of the population of individuals

		Returns
		-------
		parent : Individual
			individual that seeds the next generation
	"""

	# Get best individual just according to fitness
	idx = random.randint(0, len(parents) - 1)
	return parents[idx]

def select_new_parents(population, number_of_parents):
	"""
		Select the parent to seed the next generation.

		Parameters
		----------
		population : list
			list of instances of Individual

		Returns
		-------
		new_parents : list
			individual that seeds the next generation
	"""

	# Get best individuals ordered by fitness
	sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
	return sorted_population[0:number_of_parents]


def mutation_dsge(ind, layer, grammar):
	"""
		DSGE mutations (check DSGE for further details)


		Parameters
		----------
		ind : Individual
			Individual to mutate
		layer : dict
			layer to be mutated (DSGE genotype)
		grammar : Grammar
			Grammar instance, used to perform the initialisation and the genotype
			to phenotype mapping
	"""

	nt_keys = sorted(list(layer.keys()))
	nt_key = random.choice(nt_keys)
	nt_idx = random.randint(0, len(layer[nt_key]) - 1)
	assert nt_idx == 0

	sge_possibilities = []
	random_possibilities = []
	if len(grammar.grammar[nt_key]) > 1:
		sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) - set([layer[nt_key][nt_idx]['ge']]))
		random_possibilities.append('ge')

	if layer[nt_key][nt_idx]['ga']:
		random_possibilities.extend(['ga', 'ga'])

	if random_possibilities:
		mt_type = random.choice(random_possibilities)

		if mt_type == 'ga':
			var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
			var_type, min_val, max_val, value = layer[nt_key][nt_idx]['ga'][var_name]

			if var_type == 'int':
				while True:
					new_val = random.randint(min_val, max_val)
					if new_val != value or min_val == max_val:
						break
				ind.log_mutation(f"int {nt_key}/{var_name} {value} -> {new_val}")
			elif var_type == 'float':
				new_val = value + random.gauss(0, 0.15)
				new_val = np.clip(new_val, min_val, max_val)
				ind.log_mutation(f"float {nt_key}/{var_name} {value} -> {new_val}")

			layer[nt_key][nt_idx]['ga'][var_name] = (var_type, min_val, max_val, new_val)

		elif mt_type == 'ge':
			new_val = random.choice(sge_possibilities)
			old_val = layer[nt_key][nt_idx]['ge']
			ind.log_mutation(f"ge {nt_key} {old_val} -> {new_val}")
			layer[nt_key][nt_idx]['ge'] = new_val
			old_layer_value = deepcopy(layer)
			grammar.fix_layer_after_change(nt_key, layer)
			if layer != old_layer_value:
				ind.log_mutation_add_to_line(f"{old_layer_value}  -->  {layer}")

		else:
			return NotImplementedError


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
	ind.parent = parent.id

	# name for new individual
	ind.id = f"{gen}-{idx}"

	# mutation resets training time
	ind.reset_training()

	for module in ind.modules:
		# add-layer (duplicate or new)
		for _ in range(random.randint(1, 2)):
			if len(module.layers) < module.max_expansions and random.random() <= add_layer:
				insert_pos = random.randint(0, len(module.layers))
				if random.random() <= re_use_layer and len(module.layers):
					source_layer_index = random.randint(0, len(module.layers)-1)
					new_layer = module.layers[source_layer_index]
					layer_phenotype = grammar.decode_layer(module.module, new_layer)
					ind.log_mutation(f"copy layer {insert_pos}/{len(module.layers)} from {source_layer_index} - {layer_phenotype}")
				else:
					new_layer = grammar.initialise(module.module)
					layer_phenotype = grammar.decode_layer(module.module, new_layer)
					ind.log_mutation(f"insert layer {insert_pos}/{len(module.layers)} - {layer_phenotype}")

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
				layer_phenotype = grammar.decode_layer(module.module, module.layers[remove_idx])
				ind.log_mutation(f"remove layer {remove_idx}/{len(module.layers)} - {layer_phenotype}")
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
				mutation_dsge(ind, layer, grammar)

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
			mutation_dsge(ind, macro, grammar)

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


def do_nas_search(experiments_directory='../Experiments/', dataset='mnist', config_file='config/config.json', grammar_path='config/lenet.grammar'):  # pragma: no cover
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
		grammar_path : str
			path to the grammar file
	"""

	global USE_NETWORK_SIZE_PENALTY
	global PENALTY_CONNECTIONS_TARGET

	# load config file
	config = load_config(config_file)
	RANDOM_SEED = config['EVOLUTIONARY']['random_seed']
	NUM_GENERATIONS = config['EVOLUTIONARY']['num_generations']
	INITIAL_POUPULATION_SIZE = config['EVOLUTIONARY']['initial_population_size']
	INITIAL_INDIVIDUALS = config['EVOLUTIONARY']['initial_individuals']
	MY = config['EVOLUTIONARY']['my']
	LAMBDA = config['EVOLUTIONARY']['lambda']
	EXPERIMENT_NAME = config['EVOLUTIONARY']['experiment_name']
	RESUME = config['EVOLUTIONARY']['resume']
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

	MAX_TRAINING_TIME = config['TRAINING']['max_training_time']
	MAX_TRAINING_EPOCHS = config['TRAINING']['max_training_epochs']
	USE_NETWORK_SIZE_PENALTY = config['TRAINING']['use_network_size_penalty']
	PENALTY_CONNECTIONS_TARGET = config['TRAINING']['penalty_connections_target']
	BEST_RETEST_WITH_FINAL_TEST_SET = config['TRAINING']['best_retest_with_final_test_set']
	BEST_K_FOLDS = config['TRAINING']['best_k_folds']
	BEST_K_FOLDS_START_AT_GENERATION = config['TRAINING']['best_k_folds_start_at_generation']
	data_generator = eval(config['TRAINING']['datagen'])
	data_generator_test = eval(config['TRAINING']['datagen_test'])
	# fitness_metric = eval(config['TRAINING']['fitness_metric'])

	if not experiments_directory.endswith('/') : experiments_directory += '/'
	save_path = experiments_directory + EXPERIMENT_NAME + '/'

	# load grammar
	grammar = Grammar(grammar_path)

	# best fitness so far
	best_fitness = None
	best_individual = None

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

		# set random seeds
		if RANDOM_SEED != -1:
			random.seed(RANDOM_SEED)
			np.random.seed(RANDOM_SEED)

		# create evaluator
		cnn_eval = Evaluator(dataset, False, fitness_metric_with_size_penalty)

		# save evaluator
		pickle_evaluator(cnn_eval, save_path)

		# status variables
		last_gen = -1

		print(f'[Experiment {EXPERIMENT_NAME}] Creating the initial population of {INITIAL_POUPULATION_SIZE}')
		population = []
		for idx in range(INITIAL_POUPULATION_SIZE):
			new_individual = Individual(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, 0, idx)
			if INITIAL_INDIVIDUALS == "lenet":
				new_individual.initialise_as_lenet(grammar)
			elif INITIAL_INDIVIDUALS == "perceptron":
				new_individual.initialise_as_perceptron(grammar)
			elif INITIAL_INDIVIDUALS == "random":
				new_individual.initialise(grammar, LEVELS_BACK, REUSE_LAYER, NETWORK_STRUCTURE_INIT)
			else:
				raise RuntimeError(f"invalid value '{INITIAL_INDIVIDUALS}' of initial_individuals")
			new_individual.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
			population.append(new_individual)
		print()

	# in case there is a previous population, load it
	else:
		last_gen, cnn_eval, population, pkl_random, pkl_numpy = unpickle
		random.setstate(pkl_random)
		np.random.set_state(pkl_numpy)
		print(f'[Experiment {EXPERIMENT_NAME}] Resuming evaluation after generation {last_gen}:')

	# evaluator for K folds
	if BEST_K_FOLDS:
		k_fold_eval = Evaluator(dataset, True, fitness_metric_with_size_penalty)

	for gen in range(last_gen + 1, NUM_GENERATIONS):
		# generate offspring by mutations and evaluate population
		# population = [parent] + offspring
		if gen:
			for idx in range(LAMBDA):
				parent = select_parent(population[0:MY])
				while True:
					new_individual = mutation(parent, grammar, ADD_LAYER, REUSE_LAYER, REMOVE_LAYER, ADD_CONNECTION, REMOVE_CONNECTION, DSGE_LAYER, MACRO_LAYER, gen, idx)
					fitness = new_individual.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
					if fitness:
						break
				population.append(new_individual)

		# select parents
		new_parents = select_new_parents(population, MY)
		parent = new_parents[0]

		# update best individual
		print('[Gen %d] ' % gen, end='')
		if best_fitness is None or parent.fitness > best_fitness:
			if best_fitness:
				print('*** New best individual %s (%f acc: %f p: %d) replaces %s (%f acc: %f p: %d)' % (parent.id, parent.fitness, parent.test_accuracy, parent.parameters, best_individual, best_fitness, best_accuracy, best_parameters), end='')
			if BEST_RETEST_WITH_FINAL_TEST_SET and not parent.final_test_accuracy:
				parent.calculate_final_test_accuracy(cnn_eval)
				print('final acc: %f (acc %f)' % (parent.final_test_accuracy, parent.test_accuracy))
			if BEST_K_FOLDS and gen >= BEST_K_FOLDS_START_AT_GENERATION and not parent.k_fold_test_accuracy_average:
				parent.evaluate_with_k_fold_validation(grammar, k_fold_eval, BEST_K_FOLDS, data_generator, data_generator_test, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS)
			best_fitness = parent.fitness
			best_accuracy = parent.test_accuracy
			best_parameters = parent.parameters
			best_individual = parent.id

			# copy best individual's weights
			if os.path.isfile(Path(save_path, 'individual-%s.h5' % best_individual)):
				copyfile(Path(save_path, 'individual-%s.h5' % best_individual), Path(save_path, 'best.h5'))

			with open('%s/best_parent.pkl' % save_path, 'wb') as handle:
				pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# remove temporary files to free disk space
		if gen > 1:
			for x in range(LAMBDA):
				if os.path.isfile(Path(save_path, 'individual-%d-%d.h5' % (gen - 2, x))):
					os.remove(Path(save_path, 'individual-%d-%d.h5' % (gen - 2, x)))

		population_fitness = [ind.fitness for ind in population]
		best_in_generation_idx = np.argmax(population_fitness[MY:]) + 1 if len(population_fitness) > MY else 0
		best_in_generation = population[best_in_generation_idx]
		print('Best fitness: %f acc: %f p: %d, in generation: %f acc: %f p: %d' % (best_fitness, best_accuracy, best_parameters, best_in_generation.fitness, best_in_generation.test_accuracy, best_in_generation.parameters))

		# save population
		save_population_statistics(population, save_path, gen)
		pickle_population(gen, population, save_path)

		# keep only best as new parent
		population = [parent]

	parent = population[0]
	print('\n\n----------------------------------------------------------------------------------------------------------------------------------------')
	print('Best fitness: %s %f final: %f acc: %f p: %d' % (parent.id, parent.fitness, parent.final_test_accuracy, parent.test_accuracy, parent.parameters))
	print('\n\nPhenotype:\n')
	print(*parent.phenotype, sep="\n")
	print()
	model = load_model(Path(save_path, 'best.h5'))
	model.summary(line_length=120)
	print('\n\nEvolution history:\n')
	print(*parent.evolution_history, sep="\n")


def test_saved_model(save_path, name='best.h5'):
	# datagen_test = ImageDataGenerator()
	# evaluator = Evaluator('mnist', fitness_function)

	if not save_path.endswith('/') : save_path += '/'
	with open(Path(save_path, 'evaluator.pkl'), 'rb') as f_data:
		evaluator = pickle.load(f_data)
	X_test = evaluator.dataset['x_test']
	y_test = evaluator.dataset['y_test']

	model = load_model(Path(save_path, name))
	accuracy = test_model_with_dataset(model, X_test, y_test)
	print('Best test accuracy: %f' % accuracy)
	return model
