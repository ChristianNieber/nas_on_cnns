import unittest
import warnings
import random

class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)


	def count_unique_layers(self, modules):
		unique_layers = []
		for module in modules:
			for layer in module.layers:
				unique_layers.append(id(layer))
		
		return len(set(unique_layers))

	def count_layers(self, modules):
		return sum([len(module.layers) for module in modules])

	def create_individual(self):
		from fast_denser.utils import Individual
		from fast_denser.grammar import Grammar

		network_structure = [["features", 1, 3]]
		grammar = Grammar('tests/utilities/example.grammar')
		levels_back = {"features": 1, "classification": 1}
		network_structure_init = {"features":[2]}

		ind = Individual(network_structure, [], 'softmax', 0, 0).initialise(grammar, levels_back, 0, network_structure_init)

		print(ind.decode(grammar))

		return ind, grammar

	def create_lenet_individual(self):
		from fast_denser.utils import Individual
		from fast_denser.grammar import Grammar

		network_structure = [["features", 1, 30], ["classification", 1, 10]],
		grammar = Grammar('tests/utilities/lenet.grammar')
		levels_back = {"features": 1, "classification": 1}
		network_structure_init = {"features":[2]}

		ind = Individual(network_structure, [], 'output', 1, 0).initialise_as_lenet(grammar, levels_back, 0, network_structure_init)

		return ind, grammar

	def test_pickle_evaluator(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy
		import fast_denser.engine as engine
		import os

		random.seed(0)
		ind, grammar = self.create_individual()
		evaluator = Evaluator('cifar10', accuracy)

		if not os.path.exists('./run_0/'):
			os.makedirs('./run_0/')

		engine.pickle_evaluator(evaluator, './run_0/')

		self.assertTrue(os.path.exists('./run_0/evaluator.pkl'))


	def test_save_population(self):
		from fast_denser.utils import Individual
		from fast_denser.grammar import Grammar
		import fast_denser.engine as engine
		import os
		import random

		network_structure = [["features", 1, 3]]
		grammar = Grammar('tests/utilities/example.grammar')
		levels_back = {"features": 1, "classification": 1}
		network_structure_init = {"features":[2]}

		ind = Individual(network_structure, [], 'softmax', 0, 0)

		if not os.path.exists('./run_0/'):
			os.makedirs('./run_0/')

		engine.save_population_statistics([ind], './run_0/', 0)

		self.assertTrue(os.path.exists('./run_0/gen_0.csv'))

		engine.pickle_population([ind], ind, './run_0/')

		self.assertTrue(os.path.exists('./run_0/population.pkl'))
		self.assertTrue(os.path.exists('./run_0/parent.pkl'))
		self.assertTrue(os.path.exists('./run_0/random.pkl'))
		self.assertTrue(os.path.exists('./run_0/numpy.pkl'))

		loaded_data = engine.unpickle_population('./run_0/')

		self.assertTrue(loaded_data)


	def test_load_config(self):
		import fast_denser.engine as engine

		config = engine.load_config('tests/utilities/example_config.json')

		self.assertTrue(config)


	def test_add_layer_random(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 0, 0, 0, 0, 0, 0)

		self.assertEqual(self.count_unique_layers(ind.modules)+1, self.count_unique_layers(new_ind.modules), "Error: add layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: add layer wrong size")


	def test_add_layer_replicate(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 1, 0, 0, 0, 0, 0)

		self.assertEqual(self.count_unique_layers(ind.modules), self.count_unique_layers(new_ind.modules), "Error: duplicate layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: duplicate layer wrong size")


	def test_remove_layer(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 1, 0, 0, 0, 0)

		self.assertEqual(self.count_layers(ind.modules)-1, self.count_layers(new_ind.modules), "Error: remove layer wrong size")


	def test_mutate_ge(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 0, 0, 0, 1, 0)

		self.assertEqual(self.count_layers(ind.modules), self.count_layers(new_ind.modules), "Error: change ge parameter")

		count_ref = list()
		count_differences = 0
		total_dif = 0
		for module_idx in range(len(ind.modules)):
			for layer_idx in range(len(ind.modules[module_idx].layers)):
				total_dif += 1
				if ind.modules[module_idx].layers[layer_idx] != new_ind.modules[module_idx].layers[layer_idx]:
					if id(ind.modules[module_idx].layers[layer_idx]) not in count_ref:
						count_ref.append(id(ind.modules[module_idx].layers[layer_idx]))
						count_differences += 1

		self.assertEqual(total_dif, count_differences, "Error: change ge parameter")

	def test_keras_mapping(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy

		random.seed(0)
		ind, grammar = self.create_individual()
		evaluator = Evaluator('mnist', accuracy)

		phenotype = ind.decode(grammar)
		expected_phenotype = ("layer:conv num-filters:98 filter-shape:5 stride:2 padding:valid act:relu bias:False input:-1 "
							  "layer:conv num-filters:104 filter-shape:3 stride:1 padding:valid act:sigmoid bias:True input:0 "
							  "layer:fc act:softmax num-units:10 bias:True input:1")
		self.assertEqual(phenotype, expected_phenotype, "error in phenotype = ind.decode(grammar)")

		keras_layers = evaluator.get_layers(phenotype)
		model = evaluator.assemble_network(keras_layers, (28, 28, 1))

		model_config =  model.get_config()
		config = model.get_config()
		self.assertEqual(len(config['layers']), 8)
		self.assertEqual(len(config['input_layers']), 1)
		self.assertEqual(len(config['output_layers']), 1)
	def test_lenet(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy
		from keras.utils.layer_utils import count_params

		random.seed(0)
		ind, grammar = self.create_lenet_individual()
		evaluator = Evaluator('mnist', accuracy)

		phenotype = ind.decode(grammar)
		expected_phenotype = ("layer:conv num-filters:6 filter-shape:5 stride:1 padding:same act:relu bias:True batch-normalization:True input:-1 "
							  "layer:pool-max kernel-size:2 stride:2 padding:valid input:0 "
							  "layer:conv num-filters:16 filter-shape:5 stride:1 padding:valid act:relu bias:True batch-normalization:True input:1 "
							  "layer:pool-max kernel-size:2 stride:2 padding:valid input:2 "
							  "layer:fc act:relu num-units:120 bias:True batch-normalization:True input:3 "
							  "layer:fc act:relu num-units:84 bias:True batch-normalization:True input:4 "
							  "layer:output act:softmax num-units:10 bias:True input:5")
		self.assertEqual(phenotype, expected_phenotype, "error in phenotype = ind.decode(grammar)")

		keras_layers = evaluator.get_layers(phenotype)
		expected_keras_layers = [('conv', {'num-filters': '6', 'filter-shape': '5', 'stride': '1', 'padding': 'same', 'act': 'relu', 'bias': 'True', 'batch-normalization': 'True', 'input': '-1'}),
								 ('pool-max', {'kernel-size': '2', 'stride': '2', 'padding': 'valid', 'input': '0'}),
								 ('conv', {'num-filters': '16', 'filter-shape': '5', 'stride': '1', 'padding': 'valid', 'act': 'relu', 'bias': 'True', 'batch-normalization': 'True', 'input': '1'}),
								 ('pool-max', {'kernel-size': '2', 'stride': '2', 'padding': 'valid', 'input': '2'}),
								 ('fc', {'act': 'relu', 'num-units': '120', 'bias': 'True', 'batch-normalization': 'True', 'input': '3'}),
								 ('fc', {'act': 'relu', 'num-units': '84', 'bias': 'True', 'batch-normalization': 'True', 'input': '4'}),
								 ('output', {'act': 'softmax', 'num-units': '10', 'bias': 'True', 'input': '5'})]
		self.assertEqual(keras_layers, expected_keras_layers)

		model = evaluator.assemble_network(keras_layers, (28, 28, 1))
		self.assertEqual(model.dtype,'float32')
		self.assertEqual(model.compute_dtype,'float32')
		self.assertEqual(model.input_shape,(None, 28, 28, 1))

		config = model.get_config()
		self.assertEqual(len(config['layers']), 17)
		self.assertEqual(len(config['input_layers']), 1)
		self.assertEqual(len(config['output_layers']), 1)

		trainable_count = count_params(model.trainable_weights)
		non_trainable_count = count_params(model.non_trainable_weights)
		self.assertEqual(trainable_count, 62158)
		self.assertEqual(non_trainable_count, 452)

if __name__ == '__main__':
    unittest.main()
