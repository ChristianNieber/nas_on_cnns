# Copyright 2019 Filipe Assuncao
# modified by Christian Nieber 2023

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Alternative module implementing Fast-DENSER grammar and DSGE mutation handling

import numpy as np
import random
from logger import *
from copy import deepcopy

class Grammar:
	"""
		Dynamic Structured Grammatical Evolution (DSGE) code. F-DENSER++ uses a BNF
		grammar to define the search space, and DSGE is applied to perform the
		genotype/phenotype mapping of the inner-level of the genotype.

		Attributes
		----------
		grammar : dict
			object where the grammar is stored, and later used for initialisation,
			and decoding of the individuals.

		Methods
		-------
		get_grammar(path)
			Reads the grammar from a file
		read_grammar(path)
			Auxiliary function of the get_grammar method; loads the grammar from a file
		parse_grammar(path)
			Auxiliary function of the get_grammar method; parses the grammar to a dictionary
		_str_()
			Prints the grammar in the BNF form
		initialise(start_symbol)
			Creates a genotype, at random, starting from the input non-terminal symbol
		initialise_recursive(symbol, prev_nt, genotype)
			Auxiliary function of the initialise method; recursively expands the
			non-terminal symbol
		decode(start_symbol, genotype)
			Genotype to phenotype mapping.
		decode_recursive(symbol, read_integers, genotype, phenotype)
			Auxiliary function of the decode method; recursively applies the expansions
			that are encoded in the genotype
	"""

	def __init__(self, path):
		"""
			Parameters
			----------
			path : str
				Path to the BNF grammar file
		"""

		self.grammar = self.get_grammar(path)

	def get_grammar(self, path):
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

		raw_grammar = self.read_grammar(path)

		if raw_grammar is None:
			print('Grammar file does not exist.')
			exit(-1)

		return self.parse_grammar(raw_grammar)

	def read_grammar(self, path):
		"""
			Auxiliary function of the get_grammar method; loads the grammar from a file

			Parameters
			----------
			path : str
				Path to the BNF grammar file

			Returns
			-------
			raw_grammar : list
				list of strings, where each position is a line of the grammar file.
				Returns None in case of failure opening the file.
		"""

		try:
			with open(path, 'r') as f_in:
				raw_grammar = f_in.readlines()
				return raw_grammar
		except IOError:
			return None

	def parse_grammar(self, raw_grammar):
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

		grammar = {}
		start_symbol = None

		for rule in raw_grammar:
			rule = rule.rstrip('\n')
			if len(rule):
				[non_terminal, raw_rule_expansions] = rule.split('::=')

				rule_expansions = []
				for production_rule in raw_rule_expansions.split('|'):
					rule_expansions.append([(symbol.rstrip().lstrip().replace('<', '').replace('>', ''),
											 '<' in symbol) for symbol in
											production_rule.rstrip().lstrip().split(' ')])
				grammar[non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')] = rule_expansions

				if start_symbol is None:
					start_symbol = non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')

		return grammar

	def _str_(self):
		"""
		Prints the grammar in the BNF form
		"""

		for _key_ in sorted(self.grammar):
			productions = ''
			for production in self.grammar[_key_]:
				for symbol, terminal in production:
					if terminal:
						productions += ' <' + symbol + '>'
					else:
						productions += ' ' + symbol
				productions += ' | '
			print('<' + _key_ + '> ::=' + productions[:-3])

	def __str__(self):
		"""
		Prints the grammar in the BNF form
		"""

		print_str = ''
		for _key_ in sorted(self.grammar):
			productions = ''
			for production in self.grammar[_key_]:
				for symbol, terminal in production:
					if terminal:
						productions += ' <' + symbol + '>'
					else:
						productions += ' ' + symbol
				productions += ' | '
			print_str += '<' + _key_ + '> ::=' + productions[:-3] + '\n'

		return print_str

	def initialise_layer(self, start_symbol):
		"""
			Creates a genotype, at random, starting from the input non-terminal symbol

			Parameters
			----------
			start_symbol : str
				non-terminal symbol used as starting symbol for the grammatical expansion.

			Returns
			-------
			genotype : dict
				DSGE genotype used for the inner-level of F-DENSER++
		"""

		genotype = {}

		self.initialise_layer_recursive((start_symbol, True), None, genotype)

		return genotype

	def initialise_layer_recursive(self, symbol, prev_nt, genotype):
		"""
			Auxiliary function of the initialise method; recursively expands the
			non-terminal symbol

			Parameters
			----------
			symbol : tuple
				(non terminal symbol to expand : str, non-terminal : bool).
				Non-terminal is True in case the non-terminal symbol is a
				non-terminal, and False if the non-terminal symbol str is
				a terminal

			prev_nt: str
				non-terminal symbol used in the previous expansion

			genotype: dict
				DSGE genotype used for the inner-level of F-DENSER++

		"""

		symbol, non_terminal = symbol

		if non_terminal:
			expansion_possibility = random.randint(0, len(self.grammar[symbol]) - 1)

			if symbol not in genotype:
				genotype[symbol] = [{'ge': expansion_possibility, 'ga': {}}]
			else:
				genotype[symbol].append({'ge': expansion_possibility, 'ga': {}})

			add_reals_idx = len(genotype[symbol]) - 1
			for sym in self.grammar[symbol][expansion_possibility]:
				self.initialise_layer_recursive(sym, (symbol, add_reals_idx), genotype)
		else:
			if '[' in symbol and ']' in symbol:
				genotype_key, genotype_idx = prev_nt

				[var_name, var_type, min_val, max_val] = symbol.replace('[', '').replace(']', '').split(',')

				min_val, max_val = float(min_val), float(max_val)

				if var_type == 'int':
					value = random.randint(min_val, max_val)
				elif var_type == 'float':
					value = random.uniform(min_val, max_val)

				genotype[genotype_key][genotype_idx]['ga'][var_name] = (var_type, min_val, max_val, value)

	def decode_layer(self, start_symbol, layer_genotype):
		"""
			Genotype to phenotype mapping.

			Parameters
			----------
			start_symbol : str
				non-terminal symbol used as starting symbol for the grammatical expansion
			layer_genotype : dict
				DSGE layer genotype

			Returns
			-------
			phenotype : str
				phenotype corresponding to the input genotype
		"""

		read_integers = dict.fromkeys(list(layer_genotype.keys()), 0)
		phenotype = self.decode_layer_recursive((start_symbol, True), read_integers, layer_genotype, '')

		return phenotype.lstrip()

	def decode_layer_recursive(self, symbol, read_integers, layer_genotype, phenotype):
		"""
			Auxiliary function of the decode method; recursively applies the expansions
			that are encoded in the genotype

			Parameters
			----------
			symbol : tuple
				(non terminal symbol to expand : str, non-terminal : bool).
				Non-terminal is True in case the non-terminal symbol is a
				non-terminal, and False if the non-terminal symbol str is
				a terminal
			read_integers : dict
				integers read from genotype
			layer_genotype : dict
				DSGE layer genotype
			phenotype : str
				phenotype corresponding to the input genotype
		"""

		symbol, non_terminal = symbol

		if non_terminal:
			if symbol not in read_integers:
				read_integers[symbol] = 0
				layer_genotype[symbol] = []

			current_nt = read_integers[symbol]
			assert len(layer_genotype[symbol]) > current_nt

			expansion_integer = layer_genotype[symbol][current_nt]['ge']
			read_integers[symbol] += 1
			expansion = self.grammar[symbol][expansion_integer]

			used_terminals = []
			for sym in expansion:
				if sym[1]:
					phenotype = self.decode_layer_recursive(sym, read_integers, layer_genotype, phenotype)
				else:
					if '[' in sym[0] and ']' in sym[0]:
						[var_name, var_type, var_min, var_max] = sym[0].replace('[', '').replace(']', '').split(',')
						assert var_name in layer_genotype[symbol][current_nt]['ga']
						value = layer_genotype[symbol][current_nt]['ga'][var_name]
						if type(value) is tuple:
							value = value[-1]
						phenotype += ' %s:%s' % (var_name, value)
						used_terminals.append(var_name)
					else:
						phenotype += ' ' + sym[0]

			unused_terminals = list(set(list(layer_genotype[symbol][current_nt]['ga'].keys())) - set(used_terminals))
			assert unused_terminals == []

		return phenotype

	def fix_layer_after_change(self, start_symbol, layer_genotype):
		"""
			fix values in layer after change, e.g. generate new properties for changed layer type

			Parameters
			----------
			start_symbol : str
				non-terminal symbol used as starting symbol for the grammatical expansion
			layer_genotype : dict
				DSGE layer genotype
		"""

		read_integers = dict.fromkeys(list(layer_genotype.keys()), 0)
		self.fix_layer_after_change_recursive((start_symbol, True), read_integers, layer_genotype)

	def fix_layer_after_change_recursive(self, symbol, read_integers, layer_genotype):
		"""
			Auxiliary function of the decode method; recursively applies the expansions
			that are encoded in the genotype

			Parameters
			----------
			symbol : tuple
				(non terminal symbol to expand : str, non-terminal : bool).
				Non-terminal is True in case the non-terminal symbol is a
				non-terminal, and False if the non-terminal symbol str is
				a terminal
			read_integers : dict
				integers read from genotype
			layer_genotype : dict
				DSGE layer genotype
		"""

		symbol, non_terminal = symbol

		if non_terminal:
			if symbol not in read_integers:
				read_integers[symbol] = 0
				layer_genotype[symbol] = []

			current_nt = read_integers[symbol]
			if len(layer_genotype[symbol]) <= current_nt:
				ge_expansion_integer = random.randint(0, len(self.grammar[symbol]) - 1)
				# print(f"*** new symbol {symbol} 'ge':'{ge_expansion_integer} ***")
				layer_genotype[symbol].append({'ge': ge_expansion_integer, 'ga': {}})

			expansion_integer = layer_genotype[symbol][current_nt]['ge']
			read_integers[symbol] += 1
			expansion = self.grammar[symbol][expansion_integer]

			used_terminals = []
			for sym in expansion:
				if sym[1]:
					self.fix_layer_after_change_recursive(sym, read_integers, layer_genotype)
				else:
					if '[' in sym[0] and ']' in sym[0]:
						[var_name, var_type, var_min, var_max] = sym[0].replace('[', '').replace(']', '').split(',')
						if var_name not in layer_genotype[symbol][current_nt]['ga']:
							var_min, var_max = float(var_min), float(var_max)
							if var_type == 'int':
								value = random.randint(var_min, var_max)
							elif var_type == 'float':
								value = random.uniform(var_min, var_max)
							# print(f"*** new number {var_name}:{value} ***")
							layer_genotype[symbol][current_nt]['ga'][var_name] = (var_type, var_min, var_max, value)
						used_terminals.append(var_name)

			unused_terminals = list(set(list(layer_genotype[symbol][current_nt]['ga'].keys())) - set(used_terminals))
			if unused_terminals:
				for name in used_terminals:
					# print(f"*** remove {name} from {symbol}/{current_nt} ***")
					del layer_genotype[symbol][current_nt]['ga'][name]


class Module:
	"""
		Each of the units of the outer-level genotype

		Attributes
		----------
		module : str
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
			initialise(grammar, reuse)
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

		self.module_name = module
		self.levels_back = levels_back
		self.layers = []
		self.min_expansions = min_expansions
		self.max_expansions = max_expansions

	def initialise(self, grammar, reuse, init_max):
		"""
			Randomly creates a module

			Parameters
			----------
			grammar : Grammar
				grammar instance that stores the expansion rules

			reuse : float
				likelihood of reusing an existing layer
		"""

		num_expansions = random.choice(init_max[self.module_name])

		# Initialise layers
		for idx in range(num_expansions):
			if idx > 0 and random.random() <= reuse:
				r_idx = random.randint(0, idx - 1)
				self.layers.append(self.layers[r_idx])
			else:
				self.layers.append(grammar.initialise_layer(self.module_name))

		# Initialise connections: feed-forward and allowing skip-connections
		self.connections = {}
		for layer_idx in range(num_expansions):
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
			{'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 2.0, 64.0, 6), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 1)}}],
			 'activation-function': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 0, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'features': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'pool-type': [{'ge': 1, 'ga': {}}], 'pooling': [{'ga': {'kernel-size': ('int', 2.0, 5.0, 2), 'stride': ('int', 1.0, 3.0, 2)}, 'ge': 0}]},
			{'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 2.0, 64.0, 16), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 1)}}],
			 'activation-function': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'features': [{'ge': 1, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'pool-type': [{'ge': 1, 'ga': {}}], 'pooling': [{'ga': {'kernel-size': ('int', 2.0, 5.0, 2), 'stride': ('int', 1.0, 3.0, 2)}, 'ge': 0}]},
		]

		classification_layers_lenet = [
			{'classification': [{'ga': {'num-units': ('int', 64.0, 2048.0, 120)}, 'ge': 0}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
			{'classification': [{'ga': {'num-units': ('int', 64.0, 2048.0, 84)}, 'ge': 0}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'batch-normalization': [{'ge': 0, 'ga': {}}]},
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

def default_learning_rule_adam():
	""" default learning rule for Individual.macro initialisation """
	return [{'learning': [{'ge': 2, 'ga': {'batch_size': ('int', 50.0, 4096.0, 1024)}}],
						'adam': [{'ge': 0, 'ga': {'lr': ('float', 0.0001, 0.1, 0.0005), 'beta1': ('float', 0.5, 1.0, 0.9), 'beta2': ('float', 0.5, 1.0, 0.999)}}],
						'early-stop': [{'ge': 0, 'ga': {'early_stop': ('int', 5.0, 20.0, 8)}}]}]

#---------------------------------------------------------------------------

def mutation_dsge(ind, layer, grammar, layer_name):
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
				ind.log_mutation(f"{layer_name}: int {nt_key}/{var_name} {value} -> {new_val}")
			elif var_type == 'float':
				new_val = value + random.gauss(0, 0.15)
				new_val = np.clip(new_val, min_val, max_val)
				ind.log_mutation(f"{layer_name}: float {nt_key}/{var_name} {value} -> {new_val}")

			layer[nt_key][nt_idx]['ga'][var_name] = (var_type, min_val, max_val, new_val)

		elif mt_type == 'ge':
			new_val = random.choice(sge_possibilities)
			old_val = layer[nt_key][nt_idx]['ge']
			ind.log_mutation(f"{layer_name}: ge {nt_key} {old_val} -> {new_val}")
			layer[nt_key][nt_idx]['ge'] = new_val
			old_layer_value = deepcopy(layer)
			grammar.fix_layer_after_change(nt_key, layer)
			if layer != old_layer_value:
				ind.log_mutation_add_to_line(f"{old_layer_value}  -->  {layer}")

		else:
			return NotImplementedError
