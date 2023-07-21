from enum import Enum
import random

class Type(Enum):
	NONTERMINAL = 0
	TERMINAL = 1
	FLOAT = 2
	INT = 3
	CAT = 4

class Var:
	def __init__(self, name, type, min_value=0, max_value=0, categories=[]):
		self.name = name
		self.type = type
		if type == type.CAT:
			self.categories=categories
		elif type == Type.FLOAT or type == Type.INT:
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
		grammar : dict
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

		self.rules_list = self.read_grammar_from_file(path)

	def read_grammar_from_file(self, path):
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

		return self.parse_grammar(raw_grammar)

	@staticmethod
	def strip_symbol(symbol):
		return symbol.rstrip().lstrip().replace('<', '').replace('>', '')

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

		rules_list = {}
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
				rules_list[non_terminal] = rule_expansions

				if start_symbol is None:
					start_symbol = non_terminal

		return rules_list


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

		alternatives_list = self.rules_list[symbol]
		if len(alternatives_list) > 1:
			idx = random.randint(0, len(alternatives_list)-1)
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


