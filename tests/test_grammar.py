import unittest
import warnings


class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)

	def test_initialise(self):
		import grammar
		import random
		import numpy as np

		random.seed(0)
		np.random.seed(0)

		expected_output = [('features', [('convolution', [('layer:conv', ''), ('num-filters', 248), ('filter-shape', 5), ('stride', 2), ('padding', 'same'), ('act', 'elu'), ('bias', 'False')])])]

		# expected_output_fd = {'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, 42), 'filter-shape': ('int', 2.0, 5.0, 4), 'stride': ('int', 1.0, 3.0, 3)}}],
		# 				   'padding': [{'ge': 1, 'ga': {}}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 1, 'ga': {}}]}

		grammar = grammar.Grammar('tests/examples/example.grammar')
		output = grammar.initialise_layer('features')
		self.assertEqual(output, expected_output, "Error: initialise not equal")

	def test_decode(self):
		import grammar

		grammar = grammar.Grammar('tests/examples/example.grammar')

		start_symbol = 'features'
		layer = [('features', [('convolution', [('layer:conv', ''), ('num-filters', 242), ('filter-shape', 5), ('stride', 2), ('padding', 'valid'), ('act', 'sigmoid'), ('bias', 'True')])])]

		# layer = {'features': [{'ge': 0, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'activation-function': [{'ge': 2, 'ga': {}}],
		#			'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, 242), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 2)}}]}
		output = "layer:conv num-filters:242 filter-shape:5 stride:2 padding:valid act:sigmoid bias:True"

		phenotype = grammar.decode_layer(start_symbol, layer)

		self.assertEqual(phenotype, output, "Error: phenotypes differ")


if __name__ == '__main__':
	unittest.main()
