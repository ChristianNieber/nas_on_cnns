import unittest
import warnings


class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)

	def test_read_grammar(self):
		import grammar

		grammar = grammar.Grammar('tests/utilities/example.grammar')
		expected_output = """<activation-function> ::= act:linear |  act:relu |  act:sigmoid
<bias> ::= bias:True |  bias:False
<convolution> ::= layer:conv [num-filters,int,32,256] [filter-shape,int,2,5] [stride,int,1,3] <padding> <activation-function> <bias>
<features> ::= <convolution>
<output> ::= layer:fc num-units:10 bias:True
<padding> ::= padding:same |  padding:valid
"""
		output = grammar.__str__()
		self.assertEqual(output, expected_output, "Error: grammars differ")

	def test_read_invalid_grammar(self):
		import grammar

		with self.assertRaises(SystemExit) as cm:
			grammar = grammar.Grammar('invalid_path')
			self.assertEqual(cm.exception.code, -1, "Error: read invalid grammar")

	def test_initialise(self):
		import grammar
		import random
		import numpy as np

		random.seed(0)
		np.random.seed(0)

		expected_output = {'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, 42), 'filter-shape': ('int', 2.0, 5.0, 4), 'stride': ('int', 1.0, 3.0, 3)}}],
						   'padding': [{'ge': 1, 'ga': {}}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 1, 'ga': {}}]}

		grammar = grammar.Grammar('tests/utilities/example.grammar')

		self.assertEqual(grammar.initialise('features'), expected_output, "Error: initialise not equal")

	def test_decode(self):
		import grammar

		grammar = grammar.Grammar('tests/utilities/example.grammar')

		start_symbol = 'features'
		layer = {'features': [{'ge': 0, 'ga': {}}], 'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'activation-function': [{'ge': 2, 'ga': {}}],
					'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, 242), 'filter-shape': ('int', 2.0, 5.0, 5), 'stride': ('int', 1.0, 3.0, 2)}}]}
		output = "layer:conv num-filters:242 filter-shape:5 stride:2 padding:valid act:sigmoid bias:True"

		phenotype = grammar.decode_layer(start_symbol, layer)

		self.assertEqual(phenotype, output, "Error: phenotypes differ")


if __name__ == '__main__':
	unittest.main()