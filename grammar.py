import numpy as np
import random
from enum import Enum
from copy import copy, deepcopy
from logger import *


class Type(Enum):
	NONTERMINAL = 0
	TERMINAL = 1
	FLOAT = 2
	INT = 3
	CAT = 4


class Var:
	def __init__(self, name, value_type, min_value=None, max_value=None, categories=None):
		self.name = name
		self.type = value_type
		if value_type == value_type.CAT:
			self.categories = categories
		elif value_type == Type.FLOAT or value_type == Type.INT:
			self.min_value = min_value
			self.max_value = max_value

	# nice display in debugger
	def __repr__(self) -> str:
		result = f"Var({self.name}, {self.type}"
		if self.type == Type.CAT:
			result += f", {'/'.join(self.categories)})"
		elif self.type == Type.FLOAT or self.type == Type.INT:
			result += f", {self.min_value}, {self.max_value})"
		else:
			result += f")"
		return result


class Grammar:
	"""
		Dynamic Structured Grammatical Evolution (DSGE) code. F-DENSER++ uses a BNF
		grammar to define the search space, and DSGE is applied to perform the
		genotype/phenotype mapping of the inner-level of the genotype.

		Attributes
		----------
		rules : dict
			object where the grammar is stored, and later used for initialisation,
			and decoding of the individuals.
	"""

	def __init__(self, path):
		"""
			Parameters
			----------
			path : str
				Path to the BNF grammar file
		"""

		self.rules = Grammar.read_grammar_from_file(path)

	@staticmethod
	def read_grammar_from_file(path):
		"""
			Read the grammar from a file.

			Parameters
			----------
			path : str
				Path to the BNF grammar file

			Returns
			-------
			grammar : dict
				object where the grammar is stored, and later used for initialisation,
				and decoding of the individuals
		"""

		try:
			with open(path, 'r') as f_in:
				raw_grammar = f_in.readlines()
		except IOError:
			print('Grammar file does not exist.')
			exit(-1)

		return Grammar.parse_grammar(raw_grammar)

	@staticmethod
	def strip_symbol(symbol):
		return symbol.rstrip().lstrip().replace('<', '').replace('>', '')

	@staticmethod
	def parse_grammar(raw_grammar):
		"""
			Auxiliary function of the get_grammar method; parses the grammar to a dictionary

			Parameters
			----------
			raw_grammar : list
				list of strings, where each position is a line of the grammar file

			Returns
			-------
			grammar : dict
				object where the grammar is stored, and later used for initialisation,
				and decoding of the individuals
		"""

		rules = {}
		start_symbol = None

		for rule in raw_grammar:
			rule = rule.rstrip('\n')
			if len(rule):
				[non_terminal, raw_rule_expansions] = rule.split('::=')

				non_terminal = Grammar.strip_symbol(non_terminal)
				rule_expansions = []
				for production_rule in raw_rule_expansions.split('|'):
					vars_list = []
					for raw_symbol in production_rule.rstrip().lstrip().split(' '):
						symbol = Grammar.strip_symbol(raw_symbol)
						is_non_terminal = '<' in raw_symbol
						if is_non_terminal:
							var = Var(symbol, Type.NONTERMINAL)

						elif symbol.startswith('['):
							symbol = symbol.lstrip('[').rstrip(']')
							if ':' in symbol:
								name, categories = symbol.split(':')
								var = Var(name, Type.CAT, categories=categories.split('/'))

							else:
								name, typename, min_value, max_value = symbol.split(',')
								if typename == 'float':
									var_type = Type.FLOAT
									min_value = float(min_value)
									max_value = float(max_value)
								elif typename == 'int':
									var_type = Type.INT
									min_value = int(min_value)
									max_value = int(max_value)
								else:
									raise ValueError(f"Invalid variable type '{typename}' for variable '{name}'")
								var = Var(name, var_type, min_value, max_value)

						elif ':' in symbol:
							var = Var(symbol, Type.TERMINAL)

						vars_list.append(var)
					rule_expansions.append(vars_list)
				rules[non_terminal] = rule_expansions

				if start_symbol is None:
					start_symbol = non_terminal

		return rules

	def initialise_layer(self, symbol):
		return [(symbol, self.initialise_layer_recursive(symbol))]

	def initialise_layer_recursive(self, symbol):
		"""
			Auxiliary function of the initialise method; recursively expands the
			non-terminal symbol

			Parameters
			----------
			symbol : str
				start symbol: feature, classification, output, learning

		"""

		alternatives_list = self.rules[symbol]
		if len(alternatives_list) > 1:
			idx = random.randint(0, len(alternatives_list) - 1)
			vars_list = alternatives_list[idx]
		else:
			vars_list = alternatives_list[0]
		values_list = []
		for var in vars_list:
			if var.type == Type.NONTERMINAL:
				value = self.initialise_layer_recursive(var.name)
			elif var.type == Type.TERMINAL:
				value = ''
			elif var.type == Type.FLOAT:
				value = random.uniform(var.min_value, var.max_value)
			elif var.type == Type.INT:
				value = random.randint(var.min_value, var.max_value)
			elif var.type == Type.CAT:
				value = random.choice(var.categories)
			else:
				raise ValueError(f"Invalid var type {var.type}")
			values_list.append((var.name, value))
		return values_list

	def decode_layer(self, start_symbol, value_list):
		return Grammar.decode_value_list(value_list)

	@staticmethod
	def decode_value_list(value_list):
		phenotype = ""
		for (name, value) in value_list:
			if len(phenotype):
				phenotype += ' '
			if type(value) == list:
				phenotype += Grammar.decode_value_list(value)
			elif value == '':
				phenotype += name
			else:
				phenotype += f"{name}:{value}"
		return phenotype


class Module:
	"""
		Each of the units of the outer-level genotype

		Attributes
		----------
		module_name : str
			non-terminal symbol
		min_expansions : int
			minimum expansions of the block
		max_expansions : int
			maximum expansions of the block
		levels_back : dict
			number of previous layers a given layer can receive as input
		layers : list
			list of layers of the module
		connections : dict
			list of connections of each layer

		Methods
		-------
			initialise_module(grammar, reuse)
				Randomly creates a module
	"""

	def __init__(self, module, min_expansions, max_expansions, levels_back):
		"""
			Parameters
			----------
			module : str
				non-terminal symbol
			min_expansions : int
				minimum expansions of the block
					max_expansions : int
				maximum expansions of the block
			levels_back : int
				number of previous layers a given layer can receive as input
		"""

		self.connections = None
		self.module_name = module
		self.levels_back = levels_back
		self.layers = []

		self.min_expansions = min_expansions
		self.max_expansions = max_expansions

	def initialise_module(self, grammar, reuse, init_max):
		"""
			Randomly creates a module

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules

			reuse : float
				likelihood of reusing an existing layer
		"""

		num_layers = random.choice(init_max[self.module_name])

		# Initialise layers
		for idx in range(num_layers):
			if idx > 0 and random.random() <= reuse:
				r_idx = random.randint(0, idx - 1)
				self.layers.append(self.layers[r_idx])
			else:
				self.layers.append(grammar.initialise_layer(self.module_name))

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(num_layers):
			if layer_idx == 0:
				# the -1 layer is the input
				self.connections[layer_idx] = [-1, ]
			else:
				connection_possibilities = list(range(max(0, layer_idx - self.levels_back), layer_idx - 1))
				if len(connection_possibilities) < self.levels_back - 1:
					connection_possibilities.append(-1)

				sample_size = random.randint(0, len(connection_possibilities))

				self.connections[layer_idx] = [layer_idx - 1]
				if sample_size > 0:
					self.connections[layer_idx] += random.sample(connection_possibilities, sample_size)

	def initialise_module_as_lenet(self):
		""" Creates a pre-defined LeNet module """

		feature_layers_lenet = [
			[('features', [('convolution', [('layer:conv', ''), ('num-filters', 6), ('filter-shape', 5), ('stride', 1), ('act', 'relu'), ('padding', 'same'), ('bias', 'True'), ('batch-normalization', 'True')])])],
			[('features', [('pooling', [('layer:pooling', ''), ('kernel-size', 2), ('stride', 2), ('padding', 'valid'), ('pooling-type', 'max')])])],
			[('features', [('convolution', [('layer:conv', ''), ('num-filters', 16), ('filter-shape', 5), ('stride', 1), ('act', 'relu'), ('padding', 'valid'), ('bias', 'True'), ('batch-normalization', 'True')])])],
			[('features', [('pooling', [('layer:pooling', ''), ('kernel-size', 2), ('stride', 2), ('padding', 'valid'), ('pooling-type', 'max')])])],
		]

		classification_layers_lenet = [
			[('classification', [('layer:fc', ''), ('act', 'relu'), ('num-units', 120), ('bias', 'True'), ('batch-normalization', 'True')])],
			[('classification', [('layer:fc', ''), ('act', 'relu'), ('num-units', 84), ('bias', 'True'), ('batch-normalization', 'True')])],
		]

		if self.module_name == 'features':
			self.layers = feature_layers_lenet
		elif self.module_name == 'classification':
			self.layers = classification_layers_lenet

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(len(self.layers)):
			if layer_idx == 0:
				# the -1 layer is the input
				self.connections[layer_idx] = [-1, ]
			else:
				self.connections[layer_idx] = [layer_idx - 1]

	def initialise_module_as_perceptron(self):
		""" Creates a pre-defined Perceptron module """

		feature_layers_perceptron = []

		classification_layers_perceptron = [
			[('classification', [('layer:fc', ''), ('act', 'sigmoid'), ('num-units', 20), ('bias', 'True'), ('batch-normalization', 'False')])],
		]

		if self.module_name == 'features':
			self.layers = feature_layers_perceptron
		elif self.module_name == 'classification':
			self.layers = classification_layers_perceptron

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(len(self.layers)):
			if layer_idx == 0:
				# the -1 layer is the input
				self.connections[layer_idx] = [-1, ]
			else:
				self.connections[layer_idx] = [layer_idx - 1]


def default_learning_rule_adam():
	""" default learning rule for Individual.macro initialisation """
	return [[('learning', [('adam', [('learning:adam', ''), ('lr', 0.0005), ('beta1', 0.9), ('beta2', 0.999)])]), ('early_stop_triggered', 8), ('batch_size', 1024)]]


# ---------------------------------------------------------------------------

def mutation_dsge(ind, layer, grammar, layer_name):
	"""
		DSGE mutations (check DSGE for further details)


		Parameters
		----------
		ind : Individual
			Individual to mutate
		layer : list
			layer to be mutated (DSGE genotype)
		grammar : Grammar
			Grammar instance, used to perform the initialisation and the genotype
			to phenotype mapping
		layer_name : str
			name of layer (e.g. "<Module number>#<Layer number>") for logging
	"""

	# descend layer tree until terminal symbol encountered
	value_list = layer
	value_list_key = None
	mutable_nonterminals = []
	while True:
		(key, val) = value_list[0]
		if type(val) != list:
			break
		if len(grammar.rules[key]) > 1:
			mutable_nonterminals.append(value_list)
		value_list = val
		value_list_key = key

	alternatives_list = grammar.rules[value_list_key]
	vars_list = alternatives_list[0]
	mutable_var_indices = [i for i, var in enumerate(vars_list) if var.type.value >= Type.FLOAT.value]

	if len(mutable_var_indices) and len(mutable_nonterminals):
		if random.randint(0, 2) == 0:
			mutable_var_indices = []
		else:
			mutable_nonterminals = []

	if len(mutable_nonterminals):
		value_list = random.choice(mutable_nonterminals)
		(value_list_key, val) = value_list[0]
		key = val[0][0]
		alternatives_list = grammar.rules[value_list_key]
		list_of_choices = [varlist[0].name for varlist in alternatives_list]
		list_of_choices.remove(key)
		new_value = random.choice(list_of_choices)
		old_layer_phenotype = grammar.decode_value_list(layer)
		value_list[0] = (value_list_key, grammar.initialise_layer(new_value))
		log_mutation(f"{layer_name}: <{value_list_key}>/<{key}> -> <{new_value}>\n    {old_layer_phenotype} --> {grammar.decode_value_list(layer)}")

	elif len(mutable_var_indices):
		var_index = random.choice(mutable_var_indices)
		var = vars_list[var_index]
		var_name = var.name
		layer_value_index = -1
		for idx, item in enumerate(value_list):
			if item[0] == var_name:
				layer_value_index = idx
				break
		if layer_value_index < 0:
			log_warning(f"Variable {var_name} not present in layer {layer}")
			return

		value = value_list[layer_value_index][1]
		if var.type == Type.FLOAT:
			new_value = value + random.gauss(0, 0.15)
			new_value = np.clip(new_value, var.min_value, var.max_value)
			ind.log_mutation(f"{layer_name}: float {value_list_key}/{var_name} {value:.06f} -> {new_value:.06f}")
			value_list[layer_value_index] = (var_name, new_value)
		elif var.type == Type.INT:
			while True:
				# random.randint(var.min_value, var.max_value)
				new_value = int(value + random.gauss(0, 0.15) * (var.max_value - var.min_value))
				new_value = np.clip(new_value, var.min_value, var.max_value)
				if new_value != value or var.min_value == var.max_value:
					break
			ind.log_mutation(f"{layer_name}: int {value_list_key}/{var_name} {value} -> {new_value}")
			value_list[layer_value_index] = (var_name, new_value)
		elif var.type == Type.CAT:
			list_of_choices = var.categories.copy()
			list_of_choices.remove(value)
			new_value = random.choice(list_of_choices)
			ind.log_mutation(f"{layer_name}: {value_list_key}/{var_name} {value} -> {new_value}")
			value_list[layer_value_index] = (var_name, new_value)
		else:
			raise ValueError(f"Unexpected var type {var.type}")
