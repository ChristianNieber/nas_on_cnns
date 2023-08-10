import numpy as np
import random
from enum import Enum
from abc import ABC
from copy import copy, deepcopy
from logger import *

def format_val(val):
	"""format value for log"""
	if val == None:
		result = "None"
	elif isinstance(val, float):
		result = f"{val:.5f}"
	else:
		result = str(val)[:9]
	return result.rjust(10)


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


class StepperGrammar:
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
		self.rules = StepperGrammar.read_grammar_from_file(path)

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

		return StepperGrammar.parse_grammar(raw_grammar)

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

				non_terminal = StepperGrammar.strip_symbol(non_terminal)
				rule_expansions = []
				for production_rule in raw_rule_expansions.split('|'):
					vars_list = []
					for raw_symbol in production_rule.rstrip().lstrip().split(' '):
						symbol = StepperGrammar.strip_symbol(raw_symbol)
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

	def initialise_layer_random(self, symbol):
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
				value = self.initialise_layer_random(var.name)
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
		return StepperGrammar.decode_value_list(value_list)

	@staticmethod
	def decode_value_list(value_list):
		phenotype = ""
		for name_val_step in value_list:
			name = name_val_step[0]
			value = name_val_step[1]
			if len(phenotype):
				phenotype += ' '
			if type(value) == list:
				phenotype += StepperGrammar.decode_value_list(value)
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
			layers : list
				list of layers of the module

			Methods
			-------
				initialise_module(grammar, reuse)
					Randomly creates a module
		"""
		module_name: str
		layers: list

		def __init__(self, module, min_expansions, max_expansions):
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
			self.layers = []

			self.min_expansions = min_expansions
			self.max_expansions = max_expansions

			self.step = None
			self.previous_step = None
			self.step_history = []

		def initialise_module_random(self, grammar, init_max):
			"""
				Randomly creates a module

				Parameters
				----------
				grammar : StepperGrammar
					grammar instance that stores the expansion rules

				REUSE_LAYER : float
					likelihood of reusing an existing layer
			"""
			REUSE_LAYER = 0.15

			num_layers = random.choice(init_max[self.module_name])

			# Initialise layers
			for idx in range(num_layers):
				if idx > 0 and random.random() <= REUSE_LAYER:
					r_idx = random.randint(0, idx - 1)
					self.layers.append(self.layers[r_idx])
				else:
					self.layers.append(grammar.initialise_layer_random(self.module_name))

		def initialise_module_as_lenet(self):
			""" Creates a pre-defined LeNet module """

			feature_layers_lenet = [
				[('convolution', [('layer:conv', ''), ('num-filters', 6), ('filter-shape', 5), ('stride', 1), ('act', 'relu'), ('padding', 'same'), ('bias', 'True'), ('batch-norm', 'True')])],
				[('pooling', [('layer:pooling', ''), ('pooling-type', 'max'), ('kernel-size', 2), ('stride', 2), ('padding', 'valid')])],
				[('convolution', [('layer:conv', ''), ('num-filters', 16), ('filter-shape', 5), ('stride', 1), ('act', 'relu'), ('padding', 'valid'), ('bias', 'True'), ('batch-norm', 'True')])],
				[('pooling', [('layer:pooling', ''), ('pooling-type', 'max'), ('kernel-size', 2), ('stride', 2), ('padding', 'valid')])],
			]

			classification_layers_lenet = [
				[('layer:fc', ''), ('act', 'relu'), ('num-units', 120), ('bias', 'True'), ('batch-norm', 'True')],
				[('layer:fc', ''), ('act', 'relu'), ('num-units', 84), ('bias', 'True'), ('batch-norm', 'True')],
			]

			if self.module_name == 'features':
				self.layers = feature_layers_lenet
			elif self.module_name == 'classification':
				self.layers = classification_layers_lenet

		def initialise_module_as_perceptron(self):
			""" Creates a pre-defined Perceptron module """

			feature_layers_perceptron = []

			classification_layers_perceptron = [
				[('layer:fc', ''), ('act', 'sigmoid'), ('num-units', 20), ('bias', 'True'), ('batch-norm', 'False')],
			]

			if self.module_name == 'features':
				self.layers = feature_layers_perceptron
			elif self.module_name == 'classification':
				self.layers = classification_layers_perceptron

		@staticmethod
		def default_learning_rule_adam():
			""" default learning rule for Individual.macro initialisation """
			return [[('adam', [('learning:adam', ''), ('lr', 0.0005), ('beta1', 0.9), ('beta2', 0.999)]), ('batch_size', 1024)]]


class MutableVar:
	type: Type
	var: Var
	info_nt_key : str

	def __init__(self, type, value, step, var = None, info_module_name='', info_layer_index=0, info_nlayers=0, info_nt_key='', value_list=None):
		self.type = type
		self.value = value
		self.step = step
		self.new_value = None
		self.new_step = None
		self.var = var
		self.value_list = value_list
		self.info_module_name = info_module_name
		self.info_layer_index = info_layer_index
		self.info_nlayers = info_nlayers
		self.info_nt_key = info_nt_key

	def __repr__(self) -> str:
		result = f"MutableVar({self.info_nt_key}/{self.var.name if self.var else '<>'}, {self.value}, σ={self.step:.5f}, "
		if self.var:
			result += " " + self.var.__repr__()
		if self.value_list:
			result += f"value_list = {self.value_list}"
		if self.new_value is not None:
			result += f"new = {self.new_value} τ={self.new_step}"
		result += ', ' + self.info_string() + ')'
		return result

	def info_string(self):
		return self.info_module_name + (f"-{self.info_layer_index}/{self.info_nlayers}" if self.info_nlayers>1 else "") + ' ' + self.info_nt_key

# ---------------------------------------------------------------------------

def interval_transform(value, a, b):
	if value < a or value > b:
		y = (value-a)/(b-a)
		(integer_part, fractional_part) = np.divmod(y, 1)
		if integer_part % 2 != 0:
			fractional_part = 1 - fractional_part
		value = a + (b - a) * fractional_part
		if type(a) == int:
			value = int(round(value))
	return value


class NasStrategy():
	def __init__(self):
		self.grammar = None

	def set_grammar(self, grammar):
		self.grammar = grammar


class StepperStrategy(NasStrategy):

	def __init__(self):
		# strategy parameters
		self.EXPECTED_NUMBER_OF_PARAMETERS = 40

		self.DEFAULT_STEPWIDTH_FLOAT = 0.1
		self.DEFAULT_STEPWIDTH_INT = 0.1
		self.DEFAULT_STEPWIDTH_CAT = 0.1
		self.DEFAULT_STEPWIDTH_MODULE = 0.3
		self.DEFAULT_STEPWIDTH_MACRO = 0.3

		self.ADD_LAYER = 0.175
		self.COPY_LAYER = 0.175
		self.REMOVE_LAYER = 0.35
		self.CHANGE_TYPE = 0.3

		self.MACRO_LAYER = 0.1

	class ScanLayerInfo():
		def __init__(self, module_name, layer_index, nlayers, nonterminals_only = False):
			self.module_name = module_name
			self.layer_index = layer_index
			self.nlayers = nlayers
			self.nonterminals_only = nonterminals_only

	class MutationState():
		def __init__(self, nvars):
			self.nvars = nvars
			self.tau_global = 1.0 / np.sqrt(2 * nvars)
			self.tau_global_gaussian = self.tau_global * random.gauss(0, 1)
			self.tau_local = 1.0 / np.sqrt(2 * np.sqrt(nvars))
			self.n_mutated_vars = 0
		def set_nvars(self, nvars):
			self.nvars = nvars
		def log_normal_random(self):
			self.n_mutated_vars += 1
			return self.tau_global_gaussian + self.tau_local * random.gauss(0, 1)


	def mutate_layers(self, ind, mutation_state: MutationState):
		for module in ind.modules_including_macro:
			step = module.step
			if step == None:
				step = self.DEFAULT_STEPWIDTH_MODULE
				module.step_history = [0.0] * (ind.generation-1)
				module.step_history.append(step)
			module.previous_step = step
			tau_random_expression = mutation_state.log_normal_random()
			step = 1.0 / (1 + ((1 - step) / step) * np.exp(-tau_random_expression))
			step = interval_transform(step, 0.3333333 / mutation_state.nvars, 0.5)
			module.step = step
			module.step_history.append(step)

			u = random.uniform(0, 1)
			if u < step:  # mutate this layer?
				max_nchanges = min(2, len(module.layers) // 2) + 1  # can change 1, 2, or 3 layers, depending on the number of layers
				nchanges = int(u * max_nchanges / step) + 1

				for i in range(0, nchanges):
					probabilites = []
					actions = []
					nlayers = len(module.layers)
					if nlayers < module.max_expansions:
						probabilites.append(self.ADD_LAYER)
						actions.append('add')
						if nlayers:
							probabilites.append(self.COPY_LAYER)
							actions.append('copy')
					if nlayers > module.min_expansions:
						probabilites.append(self.REMOVE_LAYER)
						actions.append('remove')
					if nlayers:
						probabilites.append(self.CHANGE_TYPE)
						actions.append('change')

					action = actions[0] if len(actions) == 1 else random.choices(actions, weights=probabilites, k=1)[0]
					# add layer (new or copy)
					if action == 'add' or action == 'copy':
						insert_pos = random.randint(0, nlayers)
						if action == 'copy':
							source_layer_index = random.randint(0, nlayers - 1)
							new_layer = module.layers[source_layer_index]
							layer_phenotype = self.grammar.decode_layer(module.module_name, new_layer)
							ind.log_mutation(f"copy layer {module.module_name}{insert_pos}/{nlayers} from {source_layer_index} step: {module.previous_step:.5f} -> {step:.5f} - {layer_phenotype}")
						else:
							new_layer = self.grammar.initialise_layer_random(module.module_name)
							layer_phenotype = self.grammar.decode_layer(module.module_name, new_layer)
							ind.log_mutation(f"add new layer {module.module_name}{insert_pos}/{nlayers} step: {module.previous_step:.5f} -> {step:.5f} - {layer_phenotype}")
						module.layers.insert(insert_pos, new_layer)
					# remove layer
					elif action == 'remove':
						remove_idx = random.randint(0, nlayers - 1)
						layer_phenotype = self.grammar.decode_layer(module.module_name, module.layers[remove_idx])
						ind.log_mutation(f"remove layer {module.module_name}{remove_idx}/{nlayers} step: {module.previous_step:.5f} -> {step:.5f} - {layer_phenotype}")
						del module.layers[remove_idx]
					# change layer
					elif action == 'change':
						change_idx = random.randint(0, nlayers - 1)
						self.mutate_grammatical_type(ind, module, module.layers[change_idx], change_idx, nlayers, mutation_state)

	def mutate_grammatical_type(self, ind, module, layer, layer_idx, nlayers, mutation_state: MutationState):
		mutable_vars = []
		self.scan_mutable_variables_recursive(module.module_name, layer, StepperStrategy.ScanLayerInfo(module.module_name, 0, nlayers, nonterminals_only=True), mutable_vars)
		if len(mutable_vars):
			assert len(mutable_vars) == 1   # can currently mutate only one type
			mvar = mutable_vars[0]
			self.mutate_variable(mvar, mutation_state)
			index = self.write_mutated_variables_recursive(module.module_name, layer, mutable_vars, 0, nonterminals_only=True)
			ind.log_mutation(f"{mvar.info_string(): <34} {mvar.type: <25} {format_val(mvar.value): <10} -> {format_val(mvar.new_value)}, step: {format_val(module.previous_step)} -> {format_val(module.step)} : {layer}")

	def mutation(self, parent, generation=0, idx=0):
		# deep copy parent
		ind = deepcopy(parent)
		ind.parent_id = parent.id

		# name for new individual
		ind.generation = generation
		ind.id = f"{generation}-{idx}"

		# mutation resets training results
		ind.reset_training()

		# calculate τ, τ' and Nc that are used for mutating all variables of this individual
		mutation_state = StepperStrategy.MutationState(self.EXPECTED_NUMBER_OF_PARAMETERS)

		# add/copy/remove/change layers first
		self.mutate_layers(ind, mutation_state)

		# scan all layers for mutable variables
		layer_count = 0
		mutable_vars = []
		for module in ind.modules_including_macro:
			for layer_idx, layer in enumerate(module.layers):
				layer_count += 1
				self.scan_mutable_variables_recursive(module.module_name, layer, StepperStrategy.ScanLayerInfo(module.module_name, layer_idx, len(module.layers)), mutable_vars)

		# number of variables in individual are useful for some step width calculations
		nvars = len(mutable_vars) + layer_count
		mutation_state.set_nvars(nvars)

		# mutate all variables
		for mvar in mutable_vars:
			self.mutate_variable(mvar, mutation_state)

		# log changed variables
		for mvar in mutable_vars:
			if mvar.new_value != None:
				ind.log_mutation(f"{mvar.info_string(): <34} {mvar.var.name: <12} {mvar.type: <12} {format_val(mvar.value): <10} -> {format_val(mvar.new_value)}, step: {format_val(mvar.step)} -> {format_val(mvar.new_step)}")

		# write changed variables and step widths back into layers
		index = 0
		for module in ind.modules_including_macro:
			for layer_idx, layer in enumerate(module.layers):
				index = self.write_mutated_variables_recursive(module.module_name, layer, mutable_vars, index)
		assert index==len(mutable_vars)

		return ind

	def mutate_variable(self, mvar: MutableVar, mutation_state: MutationState):
		value = mvar.value
		step = mvar.step
		var = mvar.var
		type = mvar.type
		if type == Type.FLOAT:
			if step == None:
				step = self.DEFAULT_STEPWIDTH_FLOAT * (var.max_value - var.min_value)
				mvar.step = step
			step = step * np.exp(mutation_state.log_normal_random())
			gauss = random.gauss(0, 1)
			value = value + step * gauss
			value = interval_transform(value, var.min_value, var.max_value)

		elif type == Type.INT:
			if step == None:
				step = self.DEFAULT_STEPWIDTH_INT * (var.max_value - var.min_value)
				mvar.step = step
			step = step * np.exp(mutation_state.log_normal_random())
			diff = int(round(step * random.gauss(0, 1)))
			if diff:
				value += diff
				value = interval_transform(value, var.min_value, var.max_value)
			#phi = 1 - (step/mutation_state.n_mutated_vars) / (1 + np.sqrt(1 + (step/mutation_state.nvars)**2))
			#u1 = random.uniform(0.0, 1.0)
			#u2 = random.uniform(0.0, 1.0)
			#G1 = int(np.floor(np.log(1-u1) / np.log(1-phi)))
			#G2 = int(np.floor(np.log(1-u2) / np.log(1-phi)))
			#if G1 != G2:
			#	value += G1 - G2

		elif type == Type.CAT:
			if step == None:
				step = self.DEFAULT_STEPWIDTH_CAT
				mvar.step = step
			step = 1.0/(1 + ((1-step)/step) * np.exp(-mutation_state.log_normal_random()))
			step = interval_transform(step, 0.3333333 / mutation_state.nvars, 0.5)
			if random.uniform(0,1) < step:
				list_of_choices = var.categories.copy()
				list_of_choices.remove(value)
				value = random.choice(list_of_choices)

		elif type == Type.NONTERMINAL:
			# does not have its own step, this uses the module's step
			list_of_choices = mvar.value_list
			list_of_choices.remove(value)
			value = random.choice(list_of_choices)

		else:
			raise TypeError(f"mutate() not supported for {type}")

		mvar.new_step = step    # always modify step, even if value does not change!
		if value != mvar.value:
			mvar.new_value = value
			return True
		return False

	def scan_mutable_variables_recursive(self, nt_key, layer, scan_layer_info, mutable_vars):
		grammar_alternatives = self.grammar.rules[nt_key]
		for layer_idx, key_val_step in enumerate(layer):
			key = key_val_step[0]
			val = key_val_step[1]
			step = key_val_step[2] if len(key_val_step) >= 3 else None
			if type(val) == list:
				if len(grammar_alternatives) > 1 and scan_layer_info.nonterminals_only:
					nt_keys_list = [varlist[0].name for varlist in grammar_alternatives]
					assert key in nt_keys_list
					mutable_vars.append(MutableVar(Type.NONTERMINAL, key, step, None, scan_layer_info.module_name, scan_layer_info.layer_index, scan_layer_info.nlayers, nt_key, value_list=nt_keys_list))
				self.scan_mutable_variables_recursive(key, val, scan_layer_info, mutable_vars)
			else:
				var = grammar_alternatives[0][layer_idx]
				assert key == var.name, var.type != Type.NONTERMINAL
				if var.type != Type.TERMINAL and not scan_layer_info.nonterminals_only:
					mutable_vars.append(MutableVar(var.type, val, step, var, scan_layer_info.module_name, scan_layer_info.layer_index, scan_layer_info.nlayers, nt_key))

	def write_mutated_variables_recursive(self, nt_key, layer, mutable_vars, index, nonterminals_only=False):
		grammar_alternatives = self.grammar.rules[nt_key]
		for layer_idx in range(len(layer)):
			key = layer[layer_idx][0]
			val = layer[layer_idx][1]
			if type(val) == list:
				if nonterminals_only:
					key = mutable_vars[index].new_value
					val = self.grammar.initialise_layer_random(key)
					layer[layer_idx] = (key, val)
				else:
					index = self.write_mutated_variables_recursive(key, val, mutable_vars, index)
			else:
				var = grammar_alternatives[0][layer_idx]
				assert key == var.name
				assert type(val) != list
				if var.type != Type.TERMINAL and not nonterminals_only:
					if mutable_vars[index].new_step is not None:
						new_value = mutable_vars[index].new_value
						if new_value == None:
							new_value = mutable_vars[index].value
						layer[layer_idx] = (key, new_value, mutable_vars[index].new_step)
					index += 1
		return index
