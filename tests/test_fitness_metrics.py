import unittest
import warnings


class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)

	def test_accuracy(self):
		import utilities.fitness_metrics as fitness

		self.assertEqual(fitness.accuracy([0, 1], [[0, 1], [0.5, 1]]), 0.5, "Error: accuracy is wrong")

	def test_mse(self):
		import utilities.fitness_metrics as fitness

		self.assertEqual(fitness.mse([0, 1], [1, 0]), 1, "Error: mse is wrong")


if __name__ == '__main__':
	unittest.main()
