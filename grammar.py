# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import randint, uniform


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
			[non_terminal, raw_rule_expansions] = rule.rstrip('\n').split('::=')

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

	def initialise(self, start_symbol):
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

		self.initialise_recursive((start_symbol, True), None, genotype)

		return genotype

	def initialise_recursive(self, symbol, prev_nt, genotype):
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
			expansion_possibility = randint(0, len(self.grammar[symbol]) - 1)

			if symbol not in genotype:
				genotype[symbol] = [{'ge': expansion_possibility, 'ga': {}}]
			else:
				genotype[symbol].append({'ge': expansion_possibility, 'ga': {}})

			add_reals_idx = len(genotype[symbol]) - 1
			for sym in self.grammar[symbol][expansion_possibility]:
				self.initialise_recursive(sym, (symbol, add_reals_idx), genotype)
		else:
			if '[' in symbol and ']' in symbol:
				genotype_key, genotype_idx = prev_nt

				[var_name, var_type, min_val, max_val] = symbol.replace('[', '').replace(']', '').split(',')

				min_val, max_val = float(min_val), float(max_val)

				if var_type == 'int':
					value = randint(min_val, max_val)
				elif var_type == 'float':
					value = uniform(min_val, max_val)

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
				ge_expansion_integer = randint(0, len(self.grammar[symbol]) - 1)
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
								value = randint(var_min, var_max)
							elif var_type == 'float':
								value = uniform(var_min, var_max)
							# print(f"*** new number {var_name}:{value} ***")
							layer_genotype[symbol][current_nt]['ga'][var_name] = (var_type, var_min, var_max, value)
						used_terminals.append(var_name)

			unused_terminals = list(set(list(layer_genotype[symbol][current_nt]['ga'].keys())) - set(used_terminals))
			if unused_terminals:
				for name in used_terminals:
					# print(f"*** remove {name} from {symbol}/{current_nt} ***")
					del layer_genotype[symbol][current_nt]['ga'][name]