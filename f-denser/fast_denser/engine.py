# Copyright 2019 Filipe Assuncao
# Restructured 2023 Christian Nieber

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from fast_denser.grammar import Grammar
from fast_denser.utils import Evaluator, Individual
from copy import deepcopy
from os import makedirs
import pickle
import os
from shutil import copyfile
from glob import glob
import json
from fast_denser.utilities.fitness_metrics import *
from jsmin import jsmin
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from fast_denser.utilities.data_augmentation import augmentation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_population_statistics(population, save_path, gen):
	"""
		Save the current population statistics in json.
		For each individual:
			.name: name as <generation>-<index>
			.phenotype: phenotype of the individual
			.fitness: fitness of the individual
			.metrics: other evaluation metrics (e.g., loss, accuracy)
			.trainable_parameters: number of network trainable parameters
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

	with open(Path('%s/gen_%d.csv' % (save_path, gen)), 'w') as f_json:
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


def pickle_population(population, parent, save_path):
	"""
		Save the objects (pickle) necessary to later resume evolution:
		Pickled objects:
			.population
			.parent
			.random states: numpy and random
		Useful for later conducting more generations.
		Replaces the objects of the previous generation.

		Parameters
		----------
		population : list
			list of Individual instances
		parent : Individual
			fittest individual that will seed the next generation
		save_path: str
			path to the json file
	"""

	with open(Path('%s/population.pkl' % save_path), 'wb') as handle_pop:
		pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

	with open(Path('%s/parent.pkl' % save_path), 'wb') as handle_pop:
		pickle.dump(parent, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

	with open(Path('%s/random.pkl' % save_path), 'wb') as handle_random:
		pickle.dump(random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

	with open(Path('%s/numpy.pkl' % save_path), 'wb') as handle_numpy:
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
		with open(Path('%s/gen_%d.csv' % (save_path, gen))) as json_file:
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
		pickle_parent : Individual
			fittest individual of the last performed generation
		pickle_population_fitness : list
			ordered list of fitnesses of the last population of individuals
		pickle_random : tuple
			Random random state
		pickle_numpy : tuple
			Numpy random state
	"""

	csvs = glob(str(Path(save_path, '*.csv')))

	if csvs:
		csvs = [int(csv.split(os.sep)[-1].replace('gen_', '').replace('.csv', '')) for csv in csvs]
		last_generation = max(csvs)

		with open(Path(save_path, 'evaluator.pkl'), 'rb') as handle_eval:
			pickled_evaluator = pickle.load(handle_eval)

		with open(Path(save_path, 'population.pkl'), 'rb') as handle_pop:
			pickled_population = pickle.load(handle_pop)

		with open(Path(save_path, 'parent.pkl'), 'rb') as handle_parent:
			pickle_parent = pickle.load(handle_parent)

		pickle_population_fitness = [ind.fitness for ind in pickled_population]

		with open(Path(save_path, 'random.pkl'), 'rb') as handle_random:
			pickle_random = pickle.load(handle_random)

		with open(Path(save_path, 'numpy.pkl'), 'rb') as handle_numpy:
			pickle_numpy = pickle.load(handle_numpy)

		# total_epochs = get_total_epochs(save_path, last_generation)

		return last_generation, pickled_evaluator, pickled_population, pickle_parent, pickle_population_fitness, pickle_random, pickle_numpy

	else:
		return None


def select_fittest(population, population_fits):  # pragma: no cover
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

	# TODO output
	# Get best individual just according to fitness
	idx_max = np.argmax(population_fits)
	parent = population[idx_max]
	return deepcopy(parent)


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
				ind.log_mutation(f"int '{nt_key}'/'{var_name}' {value} -> {new_val}")
			elif var_type == 'float':
				new_val = value + random.gauss(0, 0.15)
				new_val = np.clip(new_val, min_val, max_val)
				ind.log_mutation(f"float '{nt_key}'/'{var_name}' {value} -> {new_val}")

			layer[nt_key][nt_idx]['ga'][var_name] = (var_type, min_val, max_val, new_val)

		elif mt_type == 'ge':
			new_val = random.choice(sge_possibilities)
			ind.log_mutation(f"ge '{nt_key}' {layer[nt_key][nt_idx]['ge']} -> {new_val}")
			layer[nt_key][nt_idx]['ge'] = new_val

		else:
			return NotImplementedError


def mutation(parent, grammar, add_layer, re_use_layer, remove_layer, add_connection, remove_connection, dsge_layer, macro_layer, gen = 0, idx = 0):
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

	#deep copy parent
	ind = deepcopy(parent)
	ind.parent = parent.name

	# name for new individual
	ind.name = f"{gen}-{idx}"

	# mutation resets training time
	ind.reset_training()

	for module in ind.modules:
		# add-layer (duplicate or new)
		for _ in range(random.randint(1, 2)):
			if len(module.layers) < module.max_expansions and random.random() <= add_layer:
				insert_pos = random.randint(0, len(module.layers))
				if random.random() <= re_use_layer:
					source_layer_index = random.randint(0, len(module.layers)-1)
					new_layer = module.layers[source_layer_index]
					layer_phenotype = grammar.decode(module.module, new_layer)
					ind.log_mutation(f"copy layer {insert_pos}/{len(module.layers)} from {source_layer_index} - {layer_phenotype}")
				else:
					new_layer = grammar.initialise(module.module)
					layer_phenotype = grammar.decode(module.module, new_layer)
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
				layer_phenotype = grammar.decode(module.module, module.layers[remove_idx])
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


def main(run, dataset, config_file, grammar_path):  # pragma: no cover
	"""
		(1+lambda)-ES

		Parameters
		----------
		run : int
			evolutionary run to perform
		dataset : str
			dataset to be solved
		config_file : str
			path to the configuration file
		grammar_path : str
			path to the grammar file
	"""

	# load config file
	config = load_config(config_file)
	RANDOM_SEEDS = config['EVOLUTIONARY']['random_seeds']
	NUMPY_SEEDS = config['EVOLUTIONARY']['numpy_seeds']
	NUM_GENERATIONS = config['EVOLUTIONARY']['num_generations']
	INITIAL_POUPULATION_SIZE = config['EVOLUTIONARY']['initial_population_size']
	LAMBDA = config['EVOLUTIONARY']['lambda']
	SAVE_PATH = config['EVOLUTIONARY']['save_path']
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
	LEVELS_BACK = config['NETWORK']['levels_back']
	NETWORK_STRUCTURE_INIT = config['NETWORK']['network_structure_init']

	MAX_TRAINING_TIME = config['TRAINING']['max_training_time']
	MAX_TRAINING_EPOCHS = config['TRAINING']['max_training_epochs']
	data_generator = eval(config['TRAINING']['datagen'])
	data_generator_test = eval(config['TRAINING']['datagen_test'])
	fitness_metric = eval(config['TRAINING']['fitness_metric'])

	save_path = '%s/run_%d/' % (SAVE_PATH, run)

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
		random.seed(RANDOM_SEEDS[run])
		np.random.seed(NUMPY_SEEDS[run])

		# create evaluator
		cnn_eval = Evaluator(dataset, fitness_metric)

		# save evaluator
		pickle_evaluator(cnn_eval, save_path)

		# status variables
		last_gen = -1
		# total_epochs = 0

	# in case there is a previous population, load it
	else:
		last_gen, cnn_eval, population, parent, population_fitness, pkl_random, pkl_numpy = unpickle
		random.setstate(pkl_random)
		np.random.set_state(pkl_numpy)

	for gen in range(last_gen + 1, NUM_GENERATIONS):
		# check the total number of epochs (stop criteria)
		# if total_epochs is not None and total_epochs >= MAX_EPOCHS:
		# 	break
		if gen == 0:
			print('[Run %d] Creating the initial population' % run)

			# create initial population
			population = [Individual(NETWORK_STRUCTURE, MACRO_STRUCTURE, OUTPUT_STRUCTURE, gen, idx).initialise_as_lenet(grammar)
						  for idx in range(INITIAL_POUPULATION_SIZE)]

			# set initial population variables and evaluate population
			population_fitness = []
			for idx, ind in enumerate(population):
				ind.training_time = 0
				population_fitness.append(ind.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS))

		else:
			# generate offspring (by mutation)
			offspring = [mutation(parent, grammar, ADD_LAYER, REUSE_LAYER, REMOVE_LAYER, ADD_CONNECTION, REMOVE_CONNECTION, DSGE_LAYER, MACRO_LAYER, gen, idx)
			             for idx in range(LAMBDA)]

			population = [parent] + offspring

			# evaluate population
			population_fitness = []
			for idx, ind in enumerate(population):
				population_fitness.append(ind.evaluate_individual(grammar, cnn_eval, data_generator, data_generator_test, save_path, MAX_TRAINING_TIME, MAX_TRAINING_EPOCHS))

		# select parent
		parent = select_fittest(population, population_fitness)

		# remove temporary files to free disk space
		if gen > 1:
			for x in range(len(population)):
				if os.path.isfile(Path(save_path, 'individual-%d-%d.h5' % (gen - 2, x))):
						os.remove(Path(save_path, 'individual-%d-%d.h5' % (gen - 2, x)))

		# update best individual
		print('[Gen %d] ' % gen, end='')
		if best_fitness is None or parent.fitness > best_fitness:
			if best_fitness:
				print('*** New best individual %s (%f) replaces %s (%f) *** ' % (parent.name, parent.fitness, best_individual, best_fitness), end='')
			best_fitness = parent.fitness
			best_individual = parent.name

			if os.path.isfile(Path(save_path, 'individual-%s.h5' % best_individual)):
				copyfile(Path(save_path, 'individual-%s.h5' % best_individual), Path(save_path, 'best.h5'))

			with open('%s/best_parent.pkl' % save_path, 'wb') as handle:
				pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print('Best fitness: %f, in generation: %f' % (best_fitness, max(population_fitness[1:]) if len(population_fitness) > 1 else max(population_fitness)))

		# save population
		save_population_statistics(population, save_path, gen)
		pickle_population(population, parent, save_path)

	# compute testing performance of the fittest network
	best_test_acc = cnn_eval.test_with_final_test_dataset(str(Path(save_path, 'best.h5')), data_generator_test)
	print('[%d] Best test accuracy: %f' % (run, best_test_acc))


def process_input(argv):  # pragma: no cover
	"""
		Maps and checks the input parameters and call the main function.

		Parameters
		----------
		argv : list
			argv from system
	"""

	dataset = None
	config_file = None
	run = 0
	grammar = None

	try:
		opts, args = getopt.getopt(argv, "hd:c:r:g:", ["dataset=", "config=", "run=", "grammar="])
	except getopt.GetoptError:
		print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
			sys.exit()

		elif opt in ("-d", "--dataset"):
			dataset = arg

		elif opt in ("-c", "--config"):
			config_file = arg

		elif opt in ("-r", "--run"):
			run = int(arg)

		elif opt in ("-g", "--grammar"):
			grammar = arg

	error = False

	# check if mandatory variables are all set
	if dataset is None:
		print('The dataset (-d) parameter is mandatory.')
		error = True

	if config_file is None:
		print('The config. file parameter (-c) is mandatory.')
		error = True

	if grammar is None:
		print('The grammar (-g) parameter is mandatory.')
		error = True

	if error:
		print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
		exit(-1)

	# check if files exist
	if not os.path.isfile(grammar):
		print('Grammar file does not exist.')
		error = True

	if not os.path.isfile(config_file):
		print('Configuration file does not exist.')
		error = True

	if not error:
		main(run, dataset, config_file, grammar)
	else:
		print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')


if __name__ == '__main__':  # pragma: no cover
	import sys, getopt

	process_input(sys.argv[1:])
